"""``lerobot-policy-server`` serves the plugin through LeRobot's ``PolicyRunner``: the runner must
admit it, reset must clear the register memory, a request at the wrong cadence must be logged, and
RTC must stay off. A stub stands in for the model; the runner is LeRobot's own."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

N_STEPS = 30  # history_stride_s = 1 s at 30 fps
STATE_DIM = 6
IMAGE_HWC = (24, 32, 3)
CAMERA = "observation.images.front"


class _StubInner:
    """``reki.RekiPolicy`` without the model: the register memory is the list of seen frames."""

    def __init__(self) -> None:
        self.policy = torch.nn.Linear(1, 1)
        self.spec = SimpleNamespace(n_action_steps=N_STEPS, prompt_fps=30.0)
        self.memory: list[np.ndarray] = []

    def reset(self) -> None:
        self.memory.clear()

    def predict(
        self,
        images: dict[str, np.ndarray],
        state: np.ndarray,
        task: str,
        stats_key: str | None = None,
    ) -> np.ndarray:
        self.memory.append(state.copy())
        return np.zeros((N_STEPS, STATE_DIM), dtype=np.float32)


def _policy() -> Any:
    from lerobot.configs.types import FeatureType, PolicyFeature

    from lerobot_policy_reki import RekiConfig, RekiPolicy

    h, w, c = IMAGE_HWC
    cfg = RekiConfig(
        input_features={
            CAMERA: PolicyFeature(type=FeatureType.VISUAL, shape=(c, h, w)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(STATE_DIM,))},
        camera_map={CAMERA: "camera_top"},
        fps=30.0,
        n_action_steps=N_STEPS,
        device="cpu",
    )
    policy = RekiPolicy(cfg, _StubInner())  # type: ignore[arg-type]
    return policy.eval()


def _runner(policy: Any, **kwargs: Any) -> Any:
    from lerobot.inference import FeatureSpec, PolicyRunner

    from lerobot_policy_reki.processor_reki import make_reki_pre_post_processors

    names = tuple(f"joint_{i}.pos" for i in range(STATE_DIM))
    return PolicyRunner(
        policy,
        *make_reki_pre_post_processors(policy.config),
        action_interval=1 / 30,
        features=(
            FeatureSpec("observation.state", (STATE_DIM,), "float32", names=names, semantics="v1"),
            FeatureSpec(CAMERA, IMAGE_HWC, "uint8", kind="rgb", semantics="rgb-v1"),
        ),
        action_feature=FeatureSpec("action", (STATE_DIM,), "float32", names=names, semantics="v1"),
        **kwargs,
    )


def _snapshot(t: float = 0.0) -> Any:
    from lerobot.inference import ObservationSnapshot

    return ObservationSnapshot(
        {
            "observation.state": np.full(STATE_DIM, t, dtype=np.float32),
            CAMERA: np.zeros(IMAGE_HWC, dtype=np.uint8),
        },
        t,
        "put the bowl on the plate",
        observation_id=f"obs{t}",
    )


@pytest.mark.cpu
def test_runner_admits_the_plugin_as_one_chunk_per_stride() -> None:
    runner = _runner(_policy())
    caps = runner.capabilities
    assert (caps.prediction_steps, caps.execution_steps) == (N_STEPS, N_STEPS)
    assert [m.value for m in caps.modes] == ["chunk"]
    assert caps.retains_session_state
    chunk = runner.predict(_snapshot())
    assert tuple(chunk.canonical_actions.shape) == (N_STEPS, STATE_DIM)
    assert chunk.execution_steps == N_STEPS


@pytest.mark.cpu
def test_rtc_is_not_offered() -> None:
    from lerobot.inference import ExecutionMode

    with pytest.raises(ValueError, match="Unsupported execution mode"):
        _runner(_policy(), modes=(ExecutionMode.RTC_GUIDED,))


@pytest.mark.cpu
def test_session_reset_clears_the_register_memory() -> None:
    policy = _policy()
    runner = _runner(policy)
    for t in range(3):
        runner.predict(_snapshot(float(t)))
    assert len(policy.reki.memory) == 3
    runner.reset(full=True)
    assert policy.reki.memory == []
    assert policy._last_predict is None


@pytest.mark.cpu
def test_a_request_inside_the_stride_is_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from lerobot_policy_reki import modeling_reki

    clock = iter([100.0, 100.2, 101.0, 102.0])
    monkeypatch.setattr(modeling_reki.time, "monotonic", lambda: next(clock))
    policy = _policy()
    runner = _runner(policy)
    with caplog.at_level(logging.WARNING, logger=modeling_reki.logger.name):
        for t in range(4):
            runner.predict(_snapshot(float(t)))
    # 0.2 s after the first request is inside the stride; the next two are a full stride apart
    assert policy.cadence_violations == 1
    assert "register memory was trained at one predict per 1.00 s" in caplog.text


@pytest.mark.cpu
def test_select_action_does_not_check_wall_clock() -> None:
    """Simulation plays faster than real time; only the served path is timed."""
    from lerobot.utils.constants import OBS_STATE

    policy = _policy()
    batch = {
        CAMERA: torch.zeros(1, 3, 24, 32),
        OBS_STATE: torch.zeros(1, STATE_DIM),
        "task": ["put the bowl on the plate"],
    }
    for _ in range(N_STEPS * 3):
        policy.select_action(batch)
    assert policy.cadence_violations == 0
    assert len(policy.reki.memory) == 3
