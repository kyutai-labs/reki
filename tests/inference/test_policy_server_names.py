"""The policy server maps a LeRobot client to the checkpoint by name."""

from __future__ import annotations

import importlib.util
import threading
import types
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.cpu

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "deploy" / "policy_server_lerobot.py"


def _server():
    spec = importlib.util.spec_from_file_location("policy_server_lerobot", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _match_client():
    return _server().match_client


STATE = ["j0.pos", "gripper.pos"]
ACTION = ["j0.pos", "gripper.pos"]


def test_actions_follow_client_motor_order() -> None:
    order = _match_client()(STATE, ACTION, ["top"], ["gripper.pos", "j0.pos"], ["top", "wrist"])
    assert [ACTION[i] for i in order] == ["gripper.pos", "j0.pos"]


@pytest.mark.parametrize(
    ("client_state", "client_cams", "match"),
    [
        (["j0.pos"], ["top"], "no state"),
        (["j0.pos", "gripper.pos", "j1.pos"], ["top"], "not the client motors"),
        (["j0.pos", "gripper.pos"], ["front"], "no camera"),
    ],
)
def test_name_mismatch_fails(client_state: list[str], client_cams: list[str], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _match_client()(STATE, ACTION, ["top"], client_state, client_cams)


def test_dropped_state_dims_need_no_client_feature() -> None:
    """The client sends only the state dims the slice reads (a zero-padded slot reads none)."""
    server = _server()
    state = ["j0.pos", "j0.vel", "gripper.pos"]
    assert server.used_state_names(state, None) == state
    used = server.used_state_names(state, [0, None, 2])
    assert used == ["j0.pos", "gripper.pos"]
    order = server.match_client(used, ACTION, ["top"], ["j0.pos", "gripper.pos"], ["top"])
    assert [ACTION[i] for i in order] == ["j0.pos", "gripper.pos"]
    with pytest.raises(ValueError, match="no state"):
        server.match_client(state, ACTION, ["top"], ["j0.pos", "gripper.pos"], ["top"])


def test_predict_zero_fills_unused_state() -> None:
    """``_predict`` builds the state in dataset order; a dim the slice drops reads 0.0."""
    seen: dict[str, np.ndarray] = {}

    class FakePolicy:
        def predict(
            self, images: dict[str, np.ndarray], state: np.ndarray, task: str, stats_key: str | None
        ) -> np.ndarray:
            seen["state"] = state
            return np.array([[0.0, 1.0]], dtype=np.float32)

    stub = types.SimpleNamespace(
        default_task="pick",
        state_names=["j0.vel", "j0.pos"],
        used_state_names=["j0.pos"],
        _camera_map={},
        _infer_lock=threading.Lock(),
        policy=FakePolicy(),
        stats_key=None,
        _client_action_order=[1, 0],
    )
    raw = {"extra": 7.0, "j0.pos": 1.5, "j0.vel": 9.0}
    actions = _server().RekiPolicyServer._predict(stub, raw)
    assert seen["state"].dtype == np.float32
    np.testing.assert_array_equal(seen["state"], [0.0, 1.5])
    np.testing.assert_array_equal(actions, [[1.0, 0.0]])
