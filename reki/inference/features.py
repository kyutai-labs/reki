"""Featurization at inference: one raw observation → the model batch, and one model action →
the dataset's native action. Mirrors what ``CombinedLeRobotDataset`` does at training time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from reki.data.robotics.normalization import normalize_feature, quantile_keys
from reki.data.robotics.prompt import clean_task_text, normalize_task, structured_prompt
from reki.data.robotics.registry import DATASETS
from reki.data.robotics.transforms import (
    axisangle_to_euler,
    gather_with_pad,
    negate_gripper_dims,
    recenter_euler_roll,
)
from reki.inference.spec import InferenceSpec
from reki.policy.constants import OBS_LANGUAGE_TOKENS, OBS_STATE
from reki.policy.train_config import TrainConfig
from reki.vlm.qwen3_5.preprocessor import role_token_id_for_camera_index


@dataclass
class EnvObs:
    """One raw environment observation: the camera images in the policy's camera order, the
    native state vector and the task text."""

    images: list[Image.Image]
    state_raw: np.ndarray
    task_text: str


def camera_role_ids(
    n_cams: int, use_camera_role_tokens: bool, camera_indices: list[int] | None = None
) -> list[int | None] | None:
    """The camera-role token ID of each image, or None when the checkpoint has no role tokens.
    ``camera_indices`` gives the canonical camera index of each image (default
    ``range(n_cams)``)."""
    if not use_camera_role_tokens:
        return None
    idxs = camera_indices if camera_indices is not None else range(n_cams)
    return [role_token_id_for_camera_index(c) for c in idxs]


def model_state(spec: InferenceSpec, state: np.ndarray) -> np.ndarray:
    """The input state in the model layout, as training maps it (``CombinedLeRobotDataset``):
    slice, axis-angle → euler, roll shift."""
    x = torch.from_numpy(state.astype(np.float32))
    if spec.state_slice is not None:
        x = gather_with_pad(x, spec.state_slice)
    if spec.state_euler_blocks:
        blocks = [(lo, hi) for lo, hi in spec.state_euler_blocks]
        x = axisangle_to_euler(x, blocks)
        if spec.state_recenter_roll:
            recenter_euler_roll(x, blocks)
    return x.numpy()


def prep_eval_state(
    state_raw: torch.Tensor,
    state_stats: dict[str, torch.Tensor],
    *,
    dataset: str,
) -> torch.Tensor:
    """Normalize one native state vector for the state MLP.

    Training negates the gripper dims of ``dataset`` after normalization
    (``DatasetSpec.gripper_flip_state``); this does the same.
    """
    out = normalize_feature(state_raw, state_stats)
    negate_gripper_dims(out, DATASETS[dataset].gripper_flip_state)
    return out


def build_inference_sample(
    obs: EnvObs,
    *,
    preprocessor: Any,
    train_cfg: TrainConfig,
    state_stats: dict[str, torch.Tensor],
    embodiment: str,
    control_mode: str,
    device: str,
    action_dim: int,
    dataset: str,
    camera_indices: list[int] | None = None,
    speed_bin: int | None = None,
    robot_type: str | None = None,
    fps: float | None = None,
) -> dict[str, Any]:
    """The batch of one observation for ``RekiModel.predict_action_chunk``, built as training
    builds a sample: prompt, images with camera-role tokens, normalized state.

    ``action_dim`` is the number of real action dims (the sampler keeps the padding at zero).
    ``robot_type`` and ``fps`` are needed for ``prompt_verbosity == 1``.
    """
    # As training: normalize_task (meta/tasks), then one clean_task_text (structured_prompt
    # cleans its instruction field).
    task_text = normalize_task(obs.task_text)
    if train_cfg.prompt_verbosity == 1:
        if robot_type is None or fps is None:
            raise ValueError(
                f"prompt_verbosity={train_cfg.prompt_verbosity} needs robot_type + fps; got "
                f"robot_type={robot_type!r}, fps={fps!r}"
            )
        task_text = structured_prompt(task_text, robot_type, fps, control_mode, speed_bin)
    else:
        task_text = clean_task_text(task_text)

    content: list[dict[str, Any]] = [{"type": "image", "image": img} for img in obs.images]
    content.append({"type": "text", "text": task_text})
    sample = preprocessor.preprocess(
        messages=[{"role": "user", "content": content}],
        camera_role_token_ids=camera_role_ids(
            len(obs.images), train_cfg.use_camera_role_tokens, camera_indices
        ),
    )
    lang_tokens = sample["input_ids"]
    vision_inputs = sample["vision_inputs"]
    batch: dict[str, Any] = {
        OBS_LANGUAGE_TOKENS: lang_tokens.to(device),
        "vision_inputs": [
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in vision_inputs.items()
            }
        ],
        "action_dim": action_dim,
    }
    if train_cfg.state_encoding == "adaln":
        state = prep_eval_state(torch.from_numpy(obs.state_raw), state_stats, dataset=dataset)
        batch[OBS_STATE] = [state.to(device)]
        batch["state_mlp_embodiment"] = [embodiment]
    return batch


def euler_to_axisangle_np(e: np.ndarray) -> np.ndarray:
    """Euler 'xyz' (radians) → axis-angle rotation vector."""
    from scipy.spatial.transform import Rotation

    return (
        Rotation.from_euler("xyz", np.asarray(e, dtype=np.float64)).as_rotvec().astype(np.float32)
    )


def unnormalize_action(action_norm: torch.Tensor, stats: dict[str, torch.Tensor]) -> np.ndarray:
    """Invert the normalization: clamp to [-1, 1] and map to [qk01, qk99]."""
    lo, hi = quantile_keys(stats)
    q01 = stats[lo].to(dtype=action_norm.dtype, device=action_norm.device)
    q99 = stats[hi].to(dtype=action_norm.dtype, device=action_norm.device)
    centered = (action_norm.clamp(-1.0, 1.0) + 1.0) / 2.0
    return (centered * (q99 - q01) + q01).cpu().numpy()


def decode_action(
    action_norm: torch.Tensor,
    stats: dict[str, torch.Tensor],
    *,
    dataset: str,
    euler_rot_dims: tuple[int, int] | None,
) -> np.ndarray:
    """Normalized model action(s) ``[..., D]`` → the dataset's native action: undo the gripper
    flip of ``dataset``, unnormalize, and convert the euler rotation block back to axis-angle."""
    action_norm = action_norm.clone()
    negate_gripper_dims(action_norm, DATASETS[dataset].gripper_flip_action)
    action = unnormalize_action(action_norm, stats)
    if euler_rot_dims is not None:
        a, b = euler_rot_dims
        action[..., a:b] = euler_to_axisangle_np(action[..., a:b])
    return action
