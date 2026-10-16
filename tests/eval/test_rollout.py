"""The rollout loop and a socket adapter against a fake sim server (no simulator, no model)."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.cpu


CAMS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
ROBOTWIN_CAMS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def _serve_one_client(srv: Any, log: list[tuple[str, bool]], dim: int) -> None:
    """A fake sim server: 5-step horizon, success on the 3rd step of each episode. Logs
    (cmd, need_obs) and answers every camera key any adapter reads."""
    from reki.eval import sim_protocol

    img = np.zeros((4, 4, 3), np.uint8)
    obs = {
        "images": dict.fromkeys(CAMS + ROBOTWIN_CAMS, img),
        "state": np.zeros(dim),
        "instruction": "x",
    }
    conn, _ = srv.accept()
    steps = 0
    while True:
        req = sim_protocol.recv(conn)
        log.append((req["cmd"], req.get("need_obs", True)))
        if req["cmd"] == "make":
            sim_protocol.send(conn, {"max_episode_steps": 5, "instruction_type": "unseen"})
        elif req["cmd"] == "reset":
            steps = 0
            sim_protocol.send(conn, obs)
        elif req["cmd"] == "step":
            steps += 1
            assert len(req["action"]) == dim
            done = {"success": steps == 3, "terminated": False, "truncated": False}
            # As the real servers: no observation on a skipped step.
            sim_protocol.send(conn, {**obs, **done} if req.get("need_obs", True) else done)
        else:
            sim_protocol.send(conn, {"ok": True})
            break
    conn.close()
    srv.close()


class _Policy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(chunk_size=2)
        self._register_cache = None

    def reset_episode(self) -> None:
        pass

    def predict_action_chunk(self, batch: dict) -> torch.Tensor:
        return torch.zeros(1, self.config.chunk_size, 32)


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """No checkpoint, model or tokenizer. Returns a settable action dim."""
    from reki.eval.rollout import loop
    from reki.policy.train_config import TrainConfig

    cfg = TrainConfig(vlm_model_id="Qwen/Qwen3.5-2B", dataset=["robocasa365"], history_k=1)
    dim = [12]
    monkeypatch.setattr(loop.TrainConfig, "from_checkpoint", staticmethod(lambda p: cfg))
    monkeypatch.setattr(loop, "eval_stats", lambda *a: (20, {}))
    stats = lambda: {"qk01": torch.zeros(dim[0]), "qk99": torch.ones(dim[0])}  # noqa: E731
    monkeypatch.setattr(loop, "_task_stats", lambda s, task: (stats(), stats()))
    monkeypatch.setattr(loop, "build_policy", lambda c, d: _Policy())
    monkeypatch.setattr(loop, "load_pretrained_weights", lambda *a, **k: None)
    monkeypatch.setattr(loop, "build_eval_preprocessor", lambda c: None)
    monkeypatch.setattr(loop, "build_inference_sample", lambda obs, **k: {})
    monkeypatch.setattr(loop, "_prompt_debug_lines", lambda *a: "")
    return dim


def _run(
    tmp_path: Path,
    adapter_cls: Any,
    task: str,
    dim: int,
    n_episodes: int,
    resume: bool,
    save_videos: bool = False,
) -> list[tuple[str, bool]]:
    from reki.eval import sim_protocol
    from reki.eval.rollout.loop import RolloutArgs, run

    srv = sim_protocol.listen("127.0.0.1", 0, None)
    log: list[tuple[str, bool]] = []
    server = threading.Thread(target=_serve_one_client, args=(srv, log, dim))
    server.start()
    adapter = adapter_cls(task, str(tmp_path), "127.0.0.1", srv.getsockname()[1])
    args = RolloutArgs(
        ckpt_dir=str(tmp_path),
        n_episodes=n_episodes,
        device="cpu",
        log_dir=str(tmp_path / "out"),
        save_videos=save_videos,
        resume=resume,
    )
    run(adapter, args)
    server.join(timeout=10)
    return log


@pytest.mark.parametrize(
    "bench, task, dim",
    [
        ("robotwin", "beat_block_hammer", 14),
        ("robocasa365", "CloseFridge", 12),
    ],
)
def test_remote_rollout_writes_results_and_resumes(
    tmp_path: Path, fake_model: list[int], bench: str, task: str, dim: int
) -> None:
    from reki.eval.rollout import remote
    from reki.eval.rollout.loop import RolloutArgs, run

    adapter_cls = {
        "robotwin": remote.RoboTwinAdapter,
        "robocasa365": remote.RoboCasa365Adapter,
    }[bench]
    fake_model[0] = dim
    log = _run(tmp_path, adapter_cls, task, dim, n_episodes=2, resume=False)
    cmds = [c for c, _ in log]
    assert cmds.count("reset") == 2 and cmds.count("step") == 6 and cmds[-1] == "close"
    # RoboTwin and RoboCasa365 skip the render inside a chunk (videos off): success at step 3
    # ends each episode inside its first chunk, so no step needs an observation.
    assert [need for c, need in log if c == "step"] == [False] * 6
    results = json.loads((tmp_path / f"out/{task}/results.json").read_text())
    assert (results["task"], results["n_episodes"], results["n_success"]) == (task, 2, 2)
    # K=1 at 20 fps: the default cadence is the 1 s history stride.
    assert results["target_episodes"] == 2 and results["n_action_steps"] == 20
    assert [e["terminal"] for e in results["episodes"]] == ["success", "success"]

    log = _run(tmp_path, adapter_cls, task, dim, n_episodes=3, resume=True)
    assert [c for c, _ in log].count("reset") == 1  # extends: only episode 2 runs
    results = json.loads((tmp_path / f"out/{task}/results.json").read_text())
    assert [e["episode"] for e in results["episodes"]] == [0, 1, 2]

    # Every unit done: nothing to run, so no connection (nothing listens on port 1).
    out = str(tmp_path / "out")
    args = RolloutArgs(ckpt_dir=str(tmp_path), n_episodes=3, device="cpu", log_dir=out, resume=True)
    run(adapter_cls(task, str(tmp_path), "127.0.0.1", 1), args)


def test_remote_rollout_observes_every_step_with_videos(
    tmp_path: Path, fake_model: list[int]
) -> None:
    """With videos on, every step asks the server for an observation."""
    from reki.eval.rollout import remote

    fake_model[0] = 12
    log = _run(tmp_path, remote.RoboCasa365Adapter, "CloseFridge", 12, 2, False, save_videos=True)
    assert [need for c, need in log if c == "step"] == [True] * 6


def test_cli_bool_flags() -> None:
    import importlib

    cli = importlib.import_module("scripts.eval.rollout")
    args = cli.parser().parse_args(["--ckpt_dir=c", "--log_dir=out", "--save_videos=0", "--resume"])
    assert args.save_videos is False and args.resume is True
    assert args.save_traj is False
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["--ckpt_dir=c", "--log_dir=out", "--use_ema=maybe"])


def test_cli_protocol_defaults() -> None:
    import importlib

    cli = importlib.import_module("scripts.eval.rollout")
    args = cli.parser().parse_args(["--ckpt_dir=c", "--log_dir=out"])
    assert args.n_episodes is None and args.save_videos is False and args.use_ema is True
    assert cli.protocol_episodes("libero", None) == 50
    assert cli.protocol_episodes("libero", "Camera Viewpoints") == 1
    assert cli.protocol_episodes("robotwin", None) == 20


def test_sim_protocol_imports_without_reki() -> None:
    """The sim servers import sim_protocol by name in their own envs (no reki installed)."""
    import subprocess
    import sys

    eval_dir = Path(__file__).resolve().parents[2] / "reki" / "eval"
    code = (
        f"import sys; sys.path.append({str(eval_dir)!r}); import sim_protocol; "
        "assert not any(m.split('.')[0] in ('reki', 'torch') for m in sys.modules)"
    )
    subprocess.run([sys.executable, "-I", "-c", code], check=True)


def test_sim_protocol_serve() -> None:
    import socket

    from reki.eval import sim_protocol

    client, server = socket.socketpair()
    errors = []

    def run() -> None:
        try:
            sim_protocol.serve(
                server,
                {"fail": lambda req: {"value": 1 / 0}, "good": lambda req: {"value": req["value"]}},
                "test",
            )
        except Exception as e:
            errors.append(e)

    thread = threading.Thread(target=run)
    thread.start()
    sim_protocol.send(client, {"cmd": "unknown"})
    assert sim_protocol.recv(client)["error"] == "ValueError: unknown cmd 'unknown'"
    sim_protocol.send(client, {"cmd": "fail"})
    assert sim_protocol.recv(client)["error"] == "ZeroDivisionError: division by zero"
    sim_protocol.send(client, {"cmd": "good", "value": 3})
    assert sim_protocol.recv(client) == {"value": 3}
    sim_protocol.send(client, {"cmd": "close"})
    assert sim_protocol.recv(client) == {"ok": True}
    thread.join(timeout=1)
    assert not thread.is_alive() and errors == []
    client.close()
    server.close()

    client, server = socket.socketpair()
    client.close()
    with pytest.raises(ConnectionError):
        sim_protocol.serve(server, {}, "test")
    server.close()


def test_each_eval_gets_its_own_triton_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Evals on one node must not share a Triton cache (rename race, Errno 39)."""
    from reki.eval.rollout.workers import set_compile_cache_dir

    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(tempfile, "tempdir", None)
    set_compile_cache_dir(str(tmp_path / "inductor"))
    first = os.environ["TRITON_CACHE_DIR"]
    set_compile_cache_dir(str(tmp_path / "inductor"))
    assert first != os.environ["TRITON_CACHE_DIR"]
    assert Path(first).is_dir() and Path(first).parent == tmp_path
