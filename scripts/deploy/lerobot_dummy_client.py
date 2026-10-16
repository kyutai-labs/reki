#!/usr/bin/env python3
"""Dummy robot client for scripts/deploy/policy_server_lerobot.py.

Replaces ``lerobot.async_inference.robot_client`` when there is no robot. It uses the same
gRPC service, the same pickled payloads (``RemotePolicyConfig``, ``TimedObservation``,
``list[TimedAction]``) and the same chunked-send helper, so it tests the wire path of the
real client: handshake, observation stream and action chunks.

The fake robot takes its state names, cameras and chunk length from the served checkpoint.
It holds a constant state and sends random uint8 camera frames, so the actions have no
meaning. It checks plumbing, shapes, cadence and the cadence gate, not policy quality.

Usage:
    pixi run python scripts/deploy/lerobot_dummy_client.py --ckpt=runs/<run>/checkpoint-N \\
        --dataset=<name> --server_address=127.0.0.1:8080 --cycles=3 --probe_gate=True
"""

from __future__ import annotations

import argparse
import pickle
import time

import grpc  # type: ignore[import-not-found]
import numpy as np
import torch
from lerobot.async_inference.helpers import RemotePolicyConfig, TimedAction, TimedObservation
from lerobot.datasets.feature_utils import hw_to_dataset_features
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.constants import OBS_STR

from reki.inference.spec import inference_spec_for
from reki.policy.checkpoint import resolve_checkpoint
from reki.policy.train_config import TrainConfig
from reki.utils.cli import str2bool


def _send(stub: object, obs: TimedObservation) -> list[TimedAction] | None:
    stub.SendObservations(  # type: ignore[attr-defined]
        send_bytes_in_chunks(
            pickle.dumps(obs), services_pb2.Observation, log_prefix="", silent=True
        )
    )
    reply = stub.GetActions(services_pb2.Empty())  # type: ignore[attr-defined]
    return None if len(reply.data) == 0 else pickle.loads(reply.data)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server_address", default="127.0.0.1:8080")
    p.add_argument("--task", default="put the bowl on the plate")
    p.add_argument("--ckpt", required=True, help="the checkpoint the server serves")
    p.add_argument(
        "--dataset", default=None, help="finetuning dataset of a raw training checkpoint"
    )
    p.add_argument("--cycles", type=int, default=3, help="How many action chunks to request.")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)
    p.add_argument(
        "--probe_gate",
        type=str2bool,
        default=False,
        help="After the first chunk, resend the same timestep: the server must gate it.",
    )
    p.add_argument("--timeout_s", type=float, default=300.0)
    args = p.parse_args()

    ckpt_dir = resolve_checkpoint(args.ckpt)
    spec, _ = inference_spec_for(ckpt_dir, TrainConfig.from_checkpoint(ckpt_dir), args.dataset)
    if spec.action_names is None or spec.camera_names is None:
        raise ValueError(f"{args.ckpt} has no robot names in its finetuning dataset")
    motors = spec.action_names
    cameras = spec.camera_names
    actions_per_chunk = spec.n_action_steps
    hw_features: dict = {m: float for m in motors}
    for cam in cameras:
        hw_features[cam] = (args.height, args.width, 3)
    lerobot_features = hw_to_dataset_features(hw_features, OBS_STR, use_video=False)
    print(f"lerobot_features: {list(lerobot_features)}")

    channel = grpc.insecure_channel(
        args.server_address, grpc_channel_options(initial_backoff=f"{1 / spec.prompt_fps:.4f}s")
    )
    stub = services_pb2_grpc.AsyncInferenceStub(channel)
    print(f"→ Ready() to {args.server_address}")
    stub.Ready(services_pb2.Empty())
    policy_config = RemotePolicyConfig(
        policy_type="reki",
        pretrained_name_or_path="served-by-the-server",
        lerobot_features=lerobot_features,
        actions_per_chunk=actions_per_chunk,
        device="cuda",
    )
    print("→ SendPolicyInstructions()")
    stub.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(policy_config)))

    rng = np.random.default_rng(0)
    state = {m: 0.1 * (i + 1) for i, m in enumerate(motors)}

    def observation(timestep: int) -> TimedObservation:
        raw: dict = dict(state)
        for cam in cameras:
            raw[cam] = rng.integers(0, 256, size=(args.height, args.width, 3), dtype=np.uint8)
        raw["task"] = args.task
        return TimedObservation(
            timestamp=time.time(), observation=raw, timestep=timestep, must_go=True
        )

    latest_action = -1
    n_chunks = 0
    gate_ok: bool | None = None
    deadline = time.time() + args.timeout_s
    while n_chunks < args.cycles:
        if time.time() > deadline:
            raise TimeoutError(f"only {n_chunks}/{args.cycles} chunks before the timeout")
        obs = observation(max(latest_action, 0))
        t0 = time.perf_counter()
        actions = _send(stub, obs)
        rtt = time.perf_counter() - t0
        if actions is None:
            print(f"   Empty (server has nothing for #{obs.get_timestep()}) — resending")
            continue
        n_chunks += 1
        steps = [a.get_timestep() for a in actions]
        assert len(actions) == actions_per_chunk, (
            f"expected {actions_per_chunk} actions, got {len(actions)}"
        )
        assert steps == list(range(steps[0], steps[0] + len(steps))), f"non-contiguous: {steps}"
        assert steps[0] == obs.get_timestep(), (
            f"chunk starts at {steps[0]}, obs was {obs.get_timestep()}"
        )
        for a in actions:
            vec = a.get_action()
            assert vec.ndim == 1 and vec.shape[0] == len(motors), f"shape {tuple(vec.shape)}"
            assert bool(torch.isfinite(vec).all()), f"non-finite action at #{a.get_timestep()}"
        print(
            f"← chunk {n_chunks}: {len(actions)} actions, timesteps {steps[0]}..{steps[-1]}, "
            f"round-trip {rtt * 1000:.0f}ms"
        )
        if args.probe_gate and gate_ok is None:
            gate_ok = _send(stub, observation(obs.get_timestep())) is None
            print(f"   gate probe: same timestep resent → {'gated' if gate_ok else 'NOT gated'}")
        latest_action = steps[-1]

    channel.close()
    if gate_ok is False:
        raise SystemExit("FAIL: the server answered an observation inside the cadence gap")
    print(f"\nOK — {n_chunks} chunks over gRPC, {actions_per_chunk} actions each.")


if __name__ == "__main__":
    main()
