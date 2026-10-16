"""Train ⇄ eval input parity per eval benchmark: ``build_inference_sample`` tokenizes to the
training item's stream (prompt, camera role tokens, image layout) and routes the state to the
same embodiment. Eval does not know the speed bin, so the test pins the speed-dropped training
variant (``speed_dropout_prob=1``)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from conftest import LEAF_SPECS, QWEN, TASK, robot_dataset
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_STATE
from PIL import Image

from reki.eval.rollout.loop import Adapter
from reki.eval.rollout.remote import robocasa365_cameras
from reki.inference.benchmarks import BENCHMARKS
from reki.inference.features import EnvObs, build_inference_sample
from reki.inference.policy import build_eval_preprocessor

pytestmark = pytest.mark.hf

SOCKET_FPS = Adapter().prompt_fps(0.0)
# Dataset -> (benchmark, prompt fps (None: the dataset's), camera indices) of the eval.
EVAL_ARGS: dict[str, tuple[str, float | None, list[int] | None]] = {
    "libero": ("libero", None, None),
    "robotwin2_clean": ("robotwin", None, None),
    "robocasa365": ("robocasa365", SOCKET_FPS, robocasa365_cameras(None)[1]),
}


@pytest.mark.parametrize("env", sorted(EVAL_ARGS))
def test_eval_input_matches_training(env: str, tiny_lerobot_leaf: Callable[..., Path]) -> None:
    from reki.policy.train_config import TrainConfig

    tiny_lerobot_leaf(env)
    common: dict[str, Any] = dict(
        vlm_model_id=QWEN,
        state_encoding="adaln",
        action_expert_num_registers=8,
        image_size=224,
        use_speed_metadata=True,
    )
    ds = robot_dataset(env, speed_dropout_prob=1.0, **common)
    member = ds[0]["group_items"][-1]
    train_cfg = TrainConfig(dataset=env, **common)

    benchmark, fps, camera_indices = EVAL_ARGS[env]
    bench = BENCHMARKS[benchmark]
    spec = LEAF_SPECS[env]
    n_cams = int(member["camera_indices"].numel())
    state_stats = ds.stats[0][OBS_STATE]
    obs = EnvObs(
        images=[Image.fromarray(np.zeros((24, 32, 3), np.uint8))] * n_cams,
        state_raw=np.zeros(state_stats["q01"].shape[0], np.float32),
        task_text=TASK,
    )
    preprocessor = build_eval_preprocessor(train_cfg)
    batch = build_inference_sample(
        obs,
        preprocessor=preprocessor,
        train_cfg=train_cfg,
        state_stats=state_stats,
        embodiment=bench.embodiment,
        control_mode=bench.control_mode,
        device="cpu",
        camera_indices=camera_indices,
        robot_type=bench.robot_type,
        fps=fps if fps is not None else spec.fps,
        action_dim=1,
        dataset=env,
    )

    train_ids, eval_ids = member[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_TOKENS]

    def text(ids: torch.Tensor) -> str:
        return preprocessor.tokenizer.decode(ids).replace("<|image_pad|>", "")

    assert torch.equal(eval_ids, train_ids), (
        f"train: {text(train_ids)!r}\neval:  {text(eval_ids)!r}"
    )
    assert batch["state_mlp_embodiment"] == [member["state_mlp_embodiment"]]
