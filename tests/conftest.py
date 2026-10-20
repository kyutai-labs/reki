"""Test tiers: every test carries exactly one tier marker, set by the module's ``pytestmark``
or, where a module mixes tiers, per test. Collection fails otherwise (markers: pyproject.toml).

cpu: pure logic, no downloads (the default gate); hf: fetches the Qwen3.5 processor/config
from the HF hub; gpu: needs a GPU (models are tiny random Qwen3.5 builds from the hub config).

    pytest -m cpu tests/        pytest -m "cpu or hf" tests/        pytest tests/

No test reads real datasets: ``REKI_DATASETS_ROOT`` is cleared for every test, and data
tests build a tiny LeRobot v3.0 leaf in ``tmp_path`` (``tiny_lerobot_leaf``).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

TIERS = ("cpu", "hf", "gpu")

QWEN = "Qwen/Qwen3.5-2B"


def robot_dataset(dataset: str, **overrides: Any) -> Any:
    """``make_robotics_dataset`` over ``dataset`` with a ``TrainConfig`` of ``overrides`` and
    deterministic sample options unless overridden."""
    from reki.data.robotics.factory import make_robotics_dataset
    from reki.policy.train_config import TrainConfig

    kw: dict[str, Any] = dict(
        dataset=dataset,
        vlm_model_id=QWEN,
        state_dropout_prob=0.0,
        speed_dropout_prob=0.0,
        embodiment_dropout_prob=0.0,
        fps_dropout_prob=0.0,
        camera_dropout_prob=0.0,
        mirror_prob=0.0,
        pin_overhead_camera=True,
        shuffle_cameras=False,
        aug_color_jitter=None,
        aug_random_crop=None,
        aug_random_scale=False,
    )
    ds, _, _ = make_robotics_dataset(TrainConfig(**{**kw, **overrides}), [dataset])
    return ds


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    bad = [
        item.nodeid
        for item in items
        if sum(item.get_closest_marker(t) is not None for t in TIERS) != 1
    ]
    if bad:
        raise pytest.UsageError(f"tests need exactly one tier marker {TIERS}: {bad}")


@pytest.fixture(autouse=True)
def _no_real_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REKI_DATASETS_ROOT", raising=False)


@dataclass(frozen=True)
class LeafSpec:
    """Raw layout of one registered dataset, as its real leaf stores it."""

    subdir: str
    robot_type: str
    fps: int
    state_dim: int
    action_dim: int
    cameras: tuple[str, ...]


LEAF_SPECS: dict[str, LeafSpec] = {
    "libero": LeafSpec(
        "HuggingFaceVLA/libero",
        "panda",
        10,
        8,
        7,
        ("observation.images.image", "observation.images.image2"),
    ),
    "robotwin2_clean": LeafSpec(
        "robotwin2_clean_lerobot/adjust_bottle",
        "aloha",
        15,
        14,
        14,
        (
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ),
    ),
    "robocasa365": LeafSpec(
        "robocasa365_lerobot_v30_collections/robocasa365/pretrain_AddIceCubes",
        "PandaOmron",
        20,
        16,
        12,
        (
            "observation.images.robot0_agentview_left",
            "observation.images.robot0_eye_in_hand",
            "observation.images.robot0_agentview_right",
        ),
    ),
}
TASK = "put the white mug on the left plate"


def _band(x: np.ndarray) -> dict[str, list]:
    """Full stats with a GLOBAL qk01/qk99 = min/max band, so every frame is inside the band."""
    out: dict[str, list] = {
        "min": x.min(0).tolist(),
        "max": x.max(0).tolist(),
        "mean": x.mean(0).tolist(),
        "std": x.std(0).tolist(),
        "count": [int(x.shape[0])],
    }
    for q in (1, 10, 50, 90, 99):
        out[f"q{q:02d}"] = np.quantile(x, q / 100, axis=0).tolist()
    out["qk01"], out["qk99"] = out["min"], out["max"]
    return out


def _write_leaf(root: Path, name: str, n_episodes: int, n_frames: int, seed: int) -> Path:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from scipy.spatial.transform import Rotation as R

    from reki.data.robotics.transforms import recenter_euler_roll

    spec = LEAF_SPECS[name]
    leaf = root / spec.subdir
    features: dict[str, Any] = {
        cam: {"dtype": "image", "shape": (24, 32, 3), "names": ["height", "width", "channel"]}
        for cam in spec.cameras
    }
    features["observation.state"] = {"dtype": "float32", "shape": (spec.state_dim,), "names": None}
    features["action"] = {"dtype": "float32", "shape": (spec.action_dim,), "names": None}
    ds = LeRobotDataset.create(
        repo_id=f"local/{name}",
        fps=spec.fps,
        features=features,
        root=leaf,
        robot_type=spec.robot_type,
        use_videos=False,
    )
    rng = np.random.default_rng(seed)
    states, actions = [], []
    for _ in range(n_episodes):
        for _ in range(n_frames):
            state = rng.uniform(-0.5, 0.5, spec.state_dim).astype(np.float32)
            action = rng.uniform(-0.5, 0.5, spec.action_dim).astype(np.float32)
            if name == "libero":
                q = rng.uniform(0.0, 0.04)
                state[6:8] = [q, -q]
                action[6] = rng.choice([-1.0, 1.0])
            frame: dict[str, Any] = {
                cam: rng.integers(0, 256, (24, 32, 3), dtype=np.uint8) for cam in spec.cameras
            }
            frame.update({"observation.state": state, "action": action, "task": TASK})
            ds.add_frame(frame)
            states.append(state)
            actions.append(action)
        ds.save_episode()
    ds.finalize()

    st, ac = np.stack(states), np.stack(actions)
    stats = json.loads((leaf / "meta" / "stats.json").read_text())
    stats["observation.state"] = _band(st)
    stats["action"] = _band(ac)
    (leaf / "meta" / "stats.json").write_text(json.dumps(stats))
    if name == "libero":
        st_e, ac_e = st.astype(np.float64), ac.astype(np.float64)
        st_e[:, 3:6] = R.from_rotvec(st_e[:, 3:6]).as_euler("xyz")
        ac_e[:, 3:6] = R.from_rotvec(ac_e[:, 3:6]).as_euler("xyz")
        sidecar = {"observation.state": _band(st_e), "action": _band(ac_e)}
        (leaf / "meta" / "norm_stats_euler.json").write_text(json.dumps(sidecar))
        sidecar["observation.state"] = _band(recenter_euler_roll(st_e))
        (leaf / "meta" / "norm_stats_euler_roll.json").write_text(json.dumps(sidecar))
    return leaf


@pytest.fixture
def tiny_lerobot_leaf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., Path]:
    """``make(name, n_episodes=2, n_frames=40, seed=0)`` writes a random-content v3.0 leaf for a
    registered dataset (``LEAF_SPECS``) under ``tmp_path`` and points the datasets root at it.
    Its stats band is the data's min/max, so every frame normalizes inside [-1, 1]."""
    monkeypatch.setenv("REKI_DATASETS_ROOT", str(tmp_path))

    def make(name: str, n_episodes: int = 2, n_frames: int = 40, seed: int = 0) -> Path:
        return _write_leaf(tmp_path, name, n_episodes, n_frames, seed)

    return make


def tiny_qwen_config() -> Any:
    """The hub Qwen3.5-2B config shrunk to 4 text layers (3 GDN + 1 full attention) and a
    2-block vision tower. Rotary/MRoPE geometry is unchanged, so the expert's head_dim floor
    (64) still applies."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(QWEN)
    t, v = cfg.text_config, cfg.vision_config
    t.hidden_size, t.intermediate_size = 512, 1024
    t.num_hidden_layers, t.layer_types = 4, t.layer_types[:4]
    t.num_attention_heads, t.num_key_value_heads = 2, 1
    t.linear_num_key_heads = t.linear_num_value_heads = 4
    v.depth, v.hidden_size, v.intermediate_size, v.num_heads = 2, 256, 512, 4
    v.out_hidden_size = t.hidden_size
    return cfg


class TinyQwenAuto:
    """Stands in for ``AutoModelForImageTextToText``: same seed → same weights on every build,
    so two loads (ours vs native) share weights like two loads of one checkpoint."""

    @staticmethod
    def from_pretrained(model_id: str, **kwargs: Any) -> Any:
        from transformers import AutoModelForImageTextToText

        if model_id != QWEN:
            raise ValueError(f"tiny Qwen stands in for {QWEN} only, got {model_id!r}")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1234)
            model = AutoModelForImageTextToText.from_config(tiny_qwen_config(), **kwargs)
            # transformers 5.8+ no longer draws matrix weights from N(0, 0.02)
            for p in model.parameters():
                if p.dim() > 1:
                    torch.nn.init.normal_(p, std=0.02)
            return model

    @staticmethod
    def from_config(config: Any, **kwargs: Any) -> Any:
        """The tiny model with other weights than ``from_pretrained`` (none under
        ``no_init_weights``): a load must overwrite them all."""
        from transformers import AutoModelForImageTextToText

        if config.name_or_path != QWEN:
            raise ValueError(f"tiny Qwen stands in for {QWEN} only, got {config.name_or_path!r}")
        return AutoModelForImageTextToText.from_config(tiny_qwen_config(), **kwargs)


@pytest.fixture
def tiny_qwen(monkeypatch: pytest.MonkeyPatch) -> type[TinyQwenAuto]:
    """Every ``Qwen3_5VLMWithExpert.from_pretrained(QWEN)`` builds the tiny random model."""
    import reki.vlm.qwen3_5.modeling as qm

    monkeypatch.setattr(qm, "AutoModelForImageTextToText", TinyQwenAuto)
    return TinyQwenAuto


TINY_EXPERT: dict[str, Any] = dict(
    action_expert_hidden_size=256,
    action_expert_intermediate_size=512,
    action_expert_num_layers=2,
    action_expert_num_heads=4,
)


@pytest.fixture
def synthetic_group(tiny_lerobot_leaf: Callable[..., Path]) -> Callable[..., tuple[Any, dict]]:
    """``make(k=3, **cfg)`` → ``(train_cfg, group)``: a K-frame register group (``group_items``)
    built by the real dataset pipeline from a tiny libero leaf (random images/state/action)."""
    from reki.data.robotics.factory import make_combined_dataset
    from reki.policy.train_config import TrainConfig

    tiny_lerobot_leaf("libero")

    def make(k: int = 3, **overrides: Any) -> tuple[Any, dict]:
        kw: dict[str, Any] = dict(
            dataset="libero",
            vlm_model_id=QWEN,
            chunk_duration_s=1,
            history_k=k,
            flow_draws=1,
            action_expert_num_registers=8,
            state_encoding="adaln",
            state_dropout_prob=0.0,
            image_size=224,
            max_cameras=2,
            aug_color_jitter=None,
            aug_random_crop=None,
            camera_dropout_prob=0.0,
            batch_size=1,
            steps=1,
            max_text_tokens=8192,
            num_workers=0,
            **TINY_EXPERT,
        )
        kw.update(overrides)
        cfg = TrainConfig(**kw)
        ds, _sampler = make_combined_dataset(cfg)
        for i in range(len(ds)):  # type: ignore[arg-type]
            item = ds[i]
            if len(item.get("group_items", [])) == k:
                return cfg, item
        raise AssertionError(f"no k_eff={k} group in the tiny leaf")

    return make
