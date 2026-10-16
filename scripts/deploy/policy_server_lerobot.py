#!/usr/bin/env python3
"""LeRobot async-inference gRPC policy server for a REKI checkpoint.

Implements the protocol of ``lerobot/async_inference`` (``transport.AsyncInference``:
Ready / SendPolicyInstructions / SendObservations / GetActions), so the stock
``lerobot.async_inference.robot_client`` drives the model without changes. The gRPC
code, observation queue and must-go filtering come from lerobot's ``PolicyServer``.
The predict step is ``reki.RekiPolicy``.

``--ckpt`` is a raw training checkpoint with ``--dataset <name>`` (its finetuning dataset; the
server reads only the dataset's ``meta/``), or an exported or Hub checkpoint, which stores its
own spec. The state, action and camera names come from the finetuning dataset. The dataset was
recorded on the same robot, so the robot client sends the same names. The server maps by name:
the dataset state in dataset order, cameras by camera name, and actions back to the client's
motor order. The client needs to send only the state dims the model reads; the others are 0.

Usage:
    pixi run python scripts/deploy/policy_server_lerobot.py --ckpt=runs/<run>/checkpoint-N \\
        --dataset=<name> --port=8080

The registers advance one frame per predict, so the predict period must equal the training
history stride (``history_stride_s``, 1 s in the released models). The server runs one predict
per ``n_action_steps`` executed actions. The client's ``--actions_per_chunk`` must equal
``n_action_steps``. ``_cadence_gap`` handles the off-by-one of the LeRobot protocol.

The protocol unpickles client messages: anyone who reaches the port can run code on this machine.
Bind 127.0.0.1 (the default) and use an SSH tunnel or a trusted network.
"""

from __future__ import annotations

import argparse
import threading
import time
from concurrent import futures
from typing import Any

import grpc  # type: ignore[import-not-found]
import numpy as np
import torch
from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation, get_logger
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

from reki import RekiPolicy
from reki.utils.cli import str2bool


def used_state_names(state_names: list[str], state_slice: list[int | None] | None) -> list[str]:
    """The dataset state names the model reads (``spec.state_slice``), in dataset order."""
    if state_slice is None:
        return list(state_names)
    used = {i for i in state_slice if i is not None}
    return [n for i, n in enumerate(state_names) if i in used]


def match_client(
    state_names: list[str],
    action_names: list[str],
    camera_names: list[str],
    client_state_names: list[str],
    client_cameras: list[str],
) -> list[int]:
    """Check the client's features against the checkpoint names, and return the permutation
    from the model's action order to the client's motor order."""
    missing = [n for n in state_names if n not in client_state_names]
    if missing:
        raise ValueError(
            f"the client sends no state {missing}. Client state: {client_state_names}; "
            f"checkpoint state: {state_names}"
        )
    if sorted(action_names) != sorted(client_state_names):
        raise ValueError(
            f"checkpoint actions {action_names} are not the client motors {client_state_names}"
        )
    missing_cams = [c for c in camera_names if c not in client_cameras]
    if missing_cams:
        raise ValueError(
            f"the client sends no camera {missing_cams}. Client cameras: {client_cameras}; "
            f"checkpoint cameras: {camera_names}"
        )
    return [action_names.index(n) for n in client_state_names]


class RekiPolicyServer(PolicyServer):
    """lerobot ``PolicyServer`` with ``RekiPolicy`` in place of a lerobot policy.

    Overrides four things: the policy setup (the REKI checkpoint, loaded at startup), the
    observation gate (register-memory cadence and episode-boundary detection), the predict
    step, and ``Ready`` (episode reset). The rest comes from lerobot unchanged: chunked
    observation receive, the maxsize-1 queue, must-go filtering and action timestamps.
    """

    prefix = "reki_policy_server"
    logger = get_logger(prefix)

    def __init__(
        self,
        config: PolicyServerConfig,
        *,
        ckpt: str,
        dataset: str | None,
        device: str,
        stats_key: str | None,
        default_task: str | None,
        idle_reset_s: float,
        compile: bool = True,
    ) -> None:
        super().__init__(config)
        self.policy = RekiPolicy.from_pretrained(
            ckpt, device=device, dataset=dataset, compile=compile
        )
        spec = self.policy.spec
        if spec.benchmark is not None:
            raise ValueError(
                f"{ckpt} is a --benchmark export ({spec.benchmark}): it has no robot joint "
                "names. Serve a checkpoint finetuned on your robot dataset (--dataset <name>)"
            )
        if spec.state_names is None or spec.action_names is None or spec.camera_names is None:
            raise ValueError(
                f"{ckpt} has no robot state/action/camera names: the finetuning dataset must "
                "name its state and action dims in meta/info.json"
            )
        self.state_names: list[str] = spec.state_names
        self.used_state_names = used_state_names(spec.state_names, spec.state_slice)
        self.action_names: list[str] = spec.action_names
        self._camera_map = dict(zip(spec.camera_names, spec.camera_keys, strict=True))
        self.stats_key = stats_key
        self.default_task = default_task
        self.idle_reset_s = idle_reset_s
        self._infer_lock = threading.Lock()
        self._episode_lock = threading.Lock()
        self._last_predict_step: int | None = None
        self._last_obs_timestep: int | None = None
        self._last_predict_wall: float | None = None
        self._client_action_order: list[int] | None = None

        self.logger.info(
            f"checkpoint: {ckpt}\n"
            f"  benchmark: {spec.benchmark} | embodiment: {spec.embodiment} | "
            f"robot: {spec.robot_type} | control mode: {spec.control_mode}\n"
            f"  state ({spec.state_dim}): {spec.state_layout}\n"
            f"  action dim: {spec.action_dim} | n_action_steps: {spec.n_action_steps} "
            f"→ predict gap {self._cadence_gap()} steps\n"
            f"  cameras: {self._camera_map}"
        )

    def _cadence_gap(self) -> int:
        """Minimum env-step gap between predicts, in the client's timestep units.

        With register memory, one predict appends one frame to the ring, so the gap must be
        the trained history stride (``n_action_steps``). The gap is ``stride - 1``. The client
        stamps an observation with the timestep of the last executed action, and drops each
        incoming action with a timestep ``<=`` that one. After a full chunk, its timestep has
        advanced by ``chunk - 1``. A gap of ``stride`` waits for a timestep that the client
        never reaches, and the server and the client deadlock.
        """
        return self.policy.n_action_steps - 1

    def _reset_episode(self, why: str) -> None:
        self.logger.info(f"episode reset ({why}) — clearing register memory")
        self.policy.reset()
        self._last_predict_step = None
        self._last_obs_timestep = None
        self._last_predict_wall = None

    def Ready(self, request: Any, context: Any):  # noqa: N802
        with self._episode_lock:
            self._reset_episode("client connected")
        return super().Ready(request, context)

    def SendPolicyInstructions(self, request: Any, context: Any):  # noqa: N802
        """Check the client's setup against this checkpoint. The server ignores the client's
        policy_type and pretrained_name_or_path: it serves the checkpoint named on its own
        command line, loaded at startup."""
        import pickle

        specs = pickle.loads(request.data)
        if not isinstance(specs, RemotePolicyConfig):
            raise TypeError(f"Policy specs must be a RemotePolicyConfig, got {type(specs)}")
        self.logger.info(
            f"client setup from {context.peer()} | policy_type={specs.policy_type!r} "
            f"pretrained={specs.pretrained_name_or_path!r} (both ignored — serving "
            f"the REKI checkpoint) | actions_per_chunk={specs.actions_per_chunk}"
        )
        self.lerobot_features = specs.lerobot_features
        self.actions_per_chunk = specs.actions_per_chunk
        self.policy_type = specs.policy_type

        n = self.policy.n_action_steps
        if specs.actions_per_chunk != n:
            raise ValueError(
                f"actions_per_chunk={specs.actions_per_chunk} but this checkpoint returns {n} "
                "actions per predict, and register memory ties one chunk to one history "
                f"step. Re-run the client with --actions_per_chunk={n}."
            )

        state_ft = self.lerobot_features.get(OBS_STATE)
        if state_ft is None:
            raise ValueError(
                f"client sent no {OBS_STATE} feature; got {sorted(self.lerobot_features)}"
            )
        cam_names = [
            k.rsplit(".", 1)[-1] for k in self.lerobot_features if k.startswith(OBS_IMAGES)
        ]
        self._client_action_order = match_client(
            self.used_state_names,
            self.action_names,
            list(self._camera_map),
            list(state_ft["names"]),
            cam_names,
        )
        return services_pb2.Empty()

    def _enqueue_observation(self, obs: TimedObservation) -> bool:
        """Cadence gate in front of lerobot's queueing: an observation that would land
        off the trained history stride is dropped, so the register ring keeps its
        spacing. GetActions then blocks on the empty queue and answers ``Empty``,
        which the client already treats as "nothing yet"."""
        step = obs.get_timestep()
        now = time.time()
        with self._episode_lock:
            if self._last_obs_timestep is not None and step < self._last_obs_timestep:
                self._reset_episode(f"timestep went backwards ({self._last_obs_timestep} → {step})")
            elif (
                self._last_predict_wall is not None
                and self.idle_reset_s > 0
                and now - self._last_predict_wall > self.idle_reset_s
            ):
                self._reset_episode(f"idle for {now - self._last_predict_wall:.1f}s")
            self._last_obs_timestep = step
            last = self._last_predict_step
        if last is not None and step - last < self._cadence_gap():
            self.logger.info(
                f"gating observation #{step} (last predict #{last}, "
                f"need +{self._cadence_gap()} steps)"
            )
            return False
        return super()._enqueue_observation(obs)

    def _predict(self, raw: dict[str, Any]) -> np.ndarray:
        task = raw.get("task") or self.default_task
        if not task:
            raise ValueError(
                "observation carries no 'task' and no --default_task was set: the prompt "
                "instruction would be empty"
            )
        missing = [k for k in self.used_state_names if k not in raw]
        if missing:
            raise KeyError(f"observation is missing state keys {missing}; has {sorted(raw)}")
        used = set(self.used_state_names)
        state = np.asarray(
            [float(raw[k]) if k in used else 0.0 for k in self.state_names], dtype=np.float32
        )
        images = {model: np.asarray(raw[client]) for client, model in self._camera_map.items()}
        with self._infer_lock:
            actions = self.policy.predict(images, state, str(task), stats_key=self.stats_key)
        assert self._client_action_order is not None
        return actions[:, self._client_action_order]

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[Any]:
        t0 = time.perf_counter()
        actions = self._predict(observation_t.get_observation())
        infer_s = time.perf_counter() - t0
        step = observation_t.get_timestep()
        with self._episode_lock:
            self._last_predict_step = step
            self._last_predict_wall = time.time()
        self.logger.info(
            f"predicted {actions.shape[0]} actions for #{step} in {infer_s * 1000:.0f}ms "
            f"| a[0]={np.round(actions[0], 2).tolist()}"
        )
        self.last_processed_obs = observation_t
        return self._time_action_chunk(
            observation_t.get_timestamp(), [torch.from_numpy(a) for a in actions], step
        )

    def warmup(self) -> None:
        """Run one throwaway predict before the server listens: FLA/Triton JIT, CUDA allocation
        and, when compiled, the compile and the default task's prefill graph. Leaves no
        register state behind."""
        spec = self.policy.spec
        images = {k: np.zeros((480, 640, 3), dtype=np.uint8) for k in spec.camera_keys}
        state = np.zeros(spec.state_dim, dtype=np.float32)
        t0 = time.perf_counter()
        self.policy.predict(images, state, self.default_task or "Warmup.", stats_key=self.stats_key)
        self.policy.reset()
        self.logger.info(f"warmup predict: {(time.perf_counter() - t0) * 1000:.0f}ms")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ckpt",
        required=True,
        help="raw training checkpoint (with --dataset), or exported checkpoint: local dir or HF id",
    )
    p.add_argument(
        "--dataset", default=None, help="finetuning dataset of a raw training checkpoint"
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--stats_key",
        default=None,
        help="Task stats key, only for a checkpoint with per-task normalization stats.",
    )
    p.add_argument(
        "--default_task",
        default=None,
        help="Instruction to use when the client sends no 'task' field.",
    )
    p.add_argument(
        "--idle_reset_s",
        type=float,
        default=5.0,
        help="Treat a gap this long since the last predict as a new episode (clears the "
        "register ring). 0 disables.",
    )
    p.add_argument("--obs_queue_timeout", type=float, default=2.0)
    p.add_argument("--warmup", type=str2bool, default=True)
    p.add_argument(
        "--compile",
        type=str2bool,
        default=True,
        help="Run the expert and the prefill as CUDA graphs (compiled during warmup).",
    )
    args = p.parse_args()

    server_cfg = PolicyServerConfig(
        host=args.host,
        port=args.port,
        fps=30,
        inference_latency=0.0,
        obs_queue_timeout=args.obs_queue_timeout,
    )
    policy_server = RekiPolicyServer(
        server_cfg,
        ckpt=args.ckpt,
        dataset=args.dataset,
        device=args.device,
        stats_key=args.stats_key,
        default_task=args.default_task,
        idle_reset_s=args.idle_reset_s,
        compile=args.compile,
    )
    fps = policy_server.policy.spec.prompt_fps
    server_cfg.fps = int(round(fps))
    policy_server.fps_tracker.target_fps = fps
    if args.warmup:
        policy_server.warmup()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{args.host}:{args.port}")
    policy_server.logger.info(f"listening on {args.host}:{args.port}")
    server.start()
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        policy_server.stop()


if __name__ == "__main__":
    main()
