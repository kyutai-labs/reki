"""``RekiPolicy``: the public inference API over an exported checkpoint."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch


def _spec(**overrides: Any) -> Any:
    from reki.inference.spec import InferenceSpec

    kw: dict[str, Any] = dict(
        benchmark="libero",
        dataset="libero",
        embodiment="single_arm_ee_euler",
        robot_type="franka",
        control_mode="delta end-effector",
        prompt_fps=10,
        chunk_size=10,
        n_action_steps=10,
        camera_keys=["camera_top", "camera_wrist_left"],
        camera_indices=None,
        state_dim=8,
        state_slice=[0, 1, 2, 3, 4, 5, None, 6],
        state_euler_blocks=[[3, 6]],
        state_recenter_roll=True,
        action_dim=7,
        euler_rot_dims=[3, 6],
        state_layout="test",
        per_task_stats=False,
        stats_keys=["shared"],
        tasks=[],
    )
    kw.update(overrides)
    return InferenceSpec(**kw)


def _policy(spec: Any, stats: Any) -> Any:
    """A ``RekiPolicy`` without a model: enough for the input checks and the state map."""
    from reki import RekiPolicy

    return RekiPolicy(cast(Any, None), None, cast(Any, None), spec, stats, "cpu")


def _obs(spec: Any) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(0)
    images = {k: rng.integers(0, 256, (24, 32, 3), dtype=np.uint8) for k in spec.camera_keys}
    return images, rng.uniform(-0.5, 0.5, spec.state_dim).astype(np.float32)


@pytest.mark.cpu
def test_predict_rejects_bad_inputs() -> None:
    spec = _spec()
    policy = _policy(spec, {"shared": ({}, {})})
    images, state = _obs(spec)
    with pytest.raises(KeyError, match="cameras"):
        policy.predict({"camera_top": images["camera_top"]}, state, "t")
    with pytest.raises(TypeError, match="uint8"):
        policy.predict(
            {**images, "camera_top": images["camera_top"].astype(np.float32)}, state, "t"
        )
    with pytest.raises(ValueError, match="HxWx3"):
        policy.predict({**images, "camera_top": images["camera_top"][..., 0]}, state, "t")
    with pytest.raises(ValueError, match="state shape"):
        policy.predict(images, state[:7], "t")
    with pytest.raises(TypeError, match="float"):
        policy.predict(images, state.astype(np.int64), "t")
    with pytest.raises(ValueError, match="shared stats set"):
        policy.predict(images, state, "t", stats_key="a")
    multi_spec = _spec(per_task_stats=True, stats_keys=["a", "b"], tasks=["a", "b"])
    multi = _policy(multi_spec, {"a": ({}, {}), "b": ({}, {})})
    with pytest.raises(ValueError, match="per-task stats; pass stats_key"):
        multi.predict(images, state, "t")
    with pytest.raises(KeyError, match="'c' not in"):
        multi.predict(images, state, "t", stats_key="c")


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("dataset", "benchmark"),
    [
        ("libero", "libero"),
        ("robotwin2_clean", "robotwin"),
        ("robocasa365", "robocasa365"),
    ],
)
def test_predict_maps_state_as_training(
    tiny_lerobot_leaf: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    dataset: str,
    benchmark: str,
) -> None:
    """``predict`` feeds the model the state ``CombinedLeRobotDataset`` trains on, bit for bit:
    the dataset state through the slice, axis-angle → euler and the roll shift."""
    from conftest import QWEN, TINY_EXPERT, robot_dataset

    import reki.inference.policy as inference
    from reki.inference.features import model_state
    from reki.inference.spec import build_inference_spec
    from reki.policy.constants import OBS_STATE
    from reki.policy.train_config import TrainConfig

    tiny_lerobot_leaf(dataset)
    cfg = TrainConfig(
        dataset=dataset,
        vlm_model_id=QWEN,
        history_k=3,
        action_expert_num_registers=8,
        state_encoding="adaln",
        max_cameras=3,
        **TINY_EXPERT,
    )
    spec, stats = build_inference_spec(cfg, benchmark=benchmark)
    train = robot_dataset(dataset)
    fed: list[np.ndarray] = []

    class _Fed(Exception):
        pass

    def build_inference_sample(obs: Any, **_: Any) -> Any:
        fed.append(obs.state_raw)
        raise _Fed

    monkeypatch.setattr(inference, "build_inference_sample", build_inference_sample)
    policy = _policy(spec, stats)
    images, _ = _obs(spec)
    rng = np.random.default_rng(0)
    for _ in range(3):
        state = rng.uniform(-1.0, 1.0, spec.state_dim).astype(np.float32)
        item = {OBS_STATE: torch.from_numpy(state.copy())}
        train._state_action_to_model_layout(item, 0)
        with pytest.raises(_Fed):
            key = spec.stats_keys[0] if spec.per_task_stats else None
            policy.predict(images, state, "t", stats_key=key)
        assert np.array_equal(fed[-1], item[OBS_STATE].numpy())
        assert np.array_equal(model_state(spec, state), item[OBS_STATE].numpy())


@pytest.mark.cpu
def test_collapse_stats() -> None:
    """One shared set only when every leaf's normalization band is bitwise identical."""
    from reki.inference.spec import collapse_stats

    def leaf(grip_hi: float, q_shift: float = 0.0) -> tuple[dict, dict]:
        state = {
            "qk01": torch.zeros(8),
            "qk99": torch.ones(8),
            "q01": torch.full((8,), q_shift),
            "q99": torch.ones(8),
            "mean": torch.full((8,), q_shift),
            "std": torch.ones(8),
        }
        action = {"qk01": torch.zeros(7), "qk99": torch.ones(7)}
        action["qk99"][6] = grip_hi
        return state, action

    shared = collapse_stats([leaf(1.0), leaf(1.0, q_shift=0.5)])
    assert shared is not None
    assert sorted(shared[0]) == ["qk01", "qk99"] and sorted(shared[1]) == ["qk01", "qk99"]
    assert collapse_stats([leaf(1.0), leaf(0.08)]) is None
    with pytest.raises(KeyError, match="qk01/qk99"):
        collapse_stats([({"min": torch.zeros(8)}, leaf(1.0)[1])])


@pytest.mark.cpu
def test_inference_spec_round_trip(tmp_path: Path) -> None:
    from reki.inference.spec import load_inference_spec, save_inference_spec

    with pytest.raises(FileNotFoundError, match="--benchmark"):
        load_inference_spec(tmp_path)
    spec = _spec(per_task_stats=True, stats_keys=["a", "b"], tasks=["a", "b"])
    stats = {
        k: (
            {"q01": torch.randn(8), "q99": torch.randn(8, dtype=torch.float64)},
            {"q01": torch.randn(7), "q99": torch.randn(7)},
        )
        for k in spec.stats_keys
    }
    save_inference_spec(tmp_path, spec, stats)
    spec2, stats2 = load_inference_spec(tmp_path)
    assert spec2 == spec
    for k in spec.stats_keys:
        for part in (0, 1):
            assert stats2[k][part].keys() == stats[k][part].keys()
            for name, t in stats[k][part].items():
                assert stats2[k][part][name].dtype == t.dtype
                assert torch.equal(stats2[k][part][name], t)
    bad = _spec(per_task_stats=True)
    save_inference_spec(tmp_path, bad, {"shared": stats["a"]})
    with pytest.raises(ValueError, match="contradicts"):
        load_inference_spec(tmp_path)
    for bad, match in (
        (_spec(state_slice=[0, 1, 2, 3, 4, 5, None, 8]), "state_slice"),
        (_spec(state_euler_blocks=[[6, 9]]), "state_euler_blocks"),
        (_spec(state_euler_blocks=[[3, 5]]), "state_euler_blocks"),
        (_spec(state_euler_blocks=[]), "state_recenter_roll"),
    ):
        save_inference_spec(tmp_path, bad, {"shared": stats["a"]})
        with pytest.raises(ValueError, match=match):
            load_inference_spec(tmp_path)
    old = json.loads((tmp_path / "reki_inference.json").read_text())
    del old["state_slice"]
    (tmp_path / "reki_inference.json").write_text(json.dumps(old))
    with pytest.raises(ValueError, match="re-export"):
        load_inference_spec(tmp_path)


@pytest.mark.gpu
@pytest.mark.usefixtures("tiny_qwen")
def test_export_from_pretrained_predict(
    synthetic_group: Callable[..., tuple[Any, dict]],
    tmp_path: Path,
) -> None:
    import safetensors.torch as st

    from reki import RekiPolicy
    from reki.data.robotics.combined_dataset import CombinedLeRobotDataset
    from reki.data.robotics.factory import make_combined_dataset
    from reki.policy.configuration import RekiConfig
    from reki.policy.modeling import RekiModel
    from reki.policy.optim import EMAModel
    from scripts.export_checkpoint import export_checkpoint

    cfg, _group = synthetic_group(k=3)
    torch.manual_seed(0)
    source = RekiModel(RekiConfig.from_train_config(cfg, device="cuda"))
    ckpt = tmp_path / "checkpoint-1"
    source.save_pretrained(ckpt)
    st.save_file(EMAModel(source, decay=0.999).state_dict(), str(ckpt / "ema.safetensors"))
    cfg.save_yaml(str(ckpt / "train_config.yaml"))
    out = tmp_path / "export"
    export_checkpoint(ckpt, out, merge_ema=True, source_rev="test", benchmark="libero")

    policy = RekiPolicy.from_pretrained(out, device="cuda")
    assert policy.n_action_steps == 10
    assert not policy.spec.per_task_stats
    combined = make_combined_dataset(cfg)[0]
    assert isinstance(combined, CombinedLeRobotDataset)
    trained = combined.stats[0]
    for exported, feat in zip(policy.stats["shared"], ("observation.state", "action"), strict=True):
        for name, t in exported.items():
            assert torch.equal(t, trained[feat][name].to(torch.float32)), (feat, name)
    images, state = _obs(policy.spec)

    def episode() -> list[np.ndarray]:
        policy.reset()
        torch.manual_seed(0)
        return [policy.predict(images, state, "pick up the bowl") for _ in range(3)]

    first = episode()
    assert policy.policy._register_cache is not None
    for actions in first:
        assert actions.shape == (10, 7) and actions.dtype == np.float32
        assert np.isfinite(actions).all()
    cache = policy.policy._register_cache
    policy.reset()
    assert policy.policy._register_cache is cache and cache.n_frames == 0
    second = episode()
    for a, b in zip(first, second, strict=True):
        assert np.array_equal(a, b)


@pytest.mark.cpu
@pytest.mark.parametrize("hub", [False, True])
def test_from_pretrained_use_ema(
    hub: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``use_ema`` overlays ``ema.safetensors`` the same way for a local dir and a Hub id."""
    import shutil
    import types

    import huggingface_hub
    import safetensors.torch as st

    import reki.inference.policy as inference
    from reki import RekiPolicy

    def tiny() -> torch.nn.Module:
        m = torch.nn.Linear(2, 2)
        m.config = types.SimpleNamespace(chunk_size=None)  # type: ignore[assignment]
        return m

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    st.save_file(
        {k: torch.zeros_like(v) for k, v in tiny().state_dict().items()},
        str(ckpt / "model.safetensors"),
    )
    st.save_file({"weight": torch.ones(2, 2)}, str(ckpt / "ema.safetensors"))

    def snapshot_download(
        repo_id: str, revision: str | None = None, ignore_patterns: list[str] | None = None
    ) -> str:
        dst = tmp_path / "hub" / str(len(list(tmp_path.glob("hub/*"))))
        shutil.copytree(ckpt, dst, ignore=shutil.ignore_patterns(*(ignore_patterns or [])))
        return str(dst)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(inference.TrainConfig, "from_checkpoint", lambda d: None)
    monkeypatch.setattr(inference, "inference_spec_for", lambda d, c, ds: (_spec(), {}))
    monkeypatch.setattr(inference, "build_policy", lambda c, device: tiny())
    monkeypatch.setattr(inference, "build_eval_preprocessor", lambda c: None)

    def weight(use_ema: bool) -> torch.Tensor:
        source = "org/reki-test" if hub else str(ckpt)
        policy = RekiPolicy.from_pretrained(source, device="cpu", use_ema=use_ema)
        return policy.policy.state_dict()["weight"]

    assert torch.equal(weight(True), torch.ones(2, 2))
    assert torch.equal(weight(False), torch.zeros(2, 2))
    (ckpt / "ema.safetensors").unlink()
    assert torch.equal(weight(True), torch.zeros(2, 2))
