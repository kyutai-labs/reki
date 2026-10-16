from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch
from conftest import QWEN, TINY_EXPERT

pytestmark = pytest.mark.cpu


def _cfg(dataset: str, **overrides: Any) -> Any:
    from reki.policy.train_config import TrainConfig

    kw: dict[str, Any] = dict(
        dataset=dataset,
        vlm_model_id=QWEN,
        chunk_duration_s=1,
        history_k=3,
        action_expert_num_registers=8,
        state_encoding="adaln",
        **TINY_EXPERT,
    )
    kw.update(overrides)
    return TrainConfig(**kw)


def _stats_equal(a: Any, b: Any) -> bool:
    return sorted(a) == sorted(b) and all(
        sorted(a[k][i]) == sorted(b[k][i])
        and all(torch.equal(a[k][i][f], b[k][i][f]) for f in a[k][i])
        for k in a
        for i in (0, 1)
    )


@pytest.mark.parametrize(
    ("dataset", "benchmark"), [("libero", "libero"), ("robotwin2_clean", "robotwin")]
)
def test_dataset_export_matches_benchmark(
    tiny_lerobot_leaf: Callable[..., Path], dataset: str, benchmark: str
) -> None:
    """--dataset derives the fields a --benchmark preset hard-codes; only the sim task names
    and the state description are preset-only."""
    from reki.inference.spec import build_inference_spec

    tiny_lerobot_leaf(dataset)
    cfg = _cfg(dataset, max_cameras=3)
    gen_spec, gen_stats = build_inference_spec(cfg, dataset=dataset)
    bench_spec, bench_stats = build_inference_spec(cfg, benchmark=benchmark)
    gen, bench = asdict(gen_spec), asdict(bench_spec)
    for key in (
        "benchmark",
        "tasks",
        "state_layout",
        "state_names",
        "action_names",
        "camera_names",
    ):
        gen.pop(key)
        bench.pop(key)
    assert gen == bench
    assert gen_spec.benchmark is None and gen_spec.tasks == []
    assert _stats_equal(gen_stats, bench_stats)


def test_dataset_export_cameras_and_layout(tiny_lerobot_leaf: Callable[..., Path]) -> None:
    """Cameras come in the canonical order training uses, with their canonical indices when
    they are not 0..n-1; the state layout lists the dataset state."""
    from reki.inference.spec import build_inference_spec

    tiny_lerobot_leaf("robocasa365")
    spec, stats = build_inference_spec(_cfg("robocasa365", max_cameras=3), dataset="robocasa365")
    assert spec.camera_keys == ["camera_top", "camera_wrist_left", "camera_side_1"]
    assert spec.camera_indices == [0, 1, 3]
    assert spec.state_dim == 16 and spec.action_dim == 12
    assert spec.state_layout.startswith("[") and spec.state_layout.count(",") == 15
    assert not spec.per_task_stats and list(stats) == ["shared"]


def test_dataset_export_per_leaf_stats(tiny_lerobot_leaf: Callable[..., Path]) -> None:
    """Leaves with different bands keep one stats set per leaf, keyed by leaf name."""
    from reki.inference.spec import build_inference_spec

    leaf = tiny_lerobot_leaf("robotwin2_clean")
    other = leaf.parent / "beat_block_hammer"
    shutil.copytree(leaf, other)
    stats_p = other / "meta" / "stats.json"
    raw = json.loads(stats_p.read_text())
    raw["action"]["qk99"] = [v + 0.5 for v in raw["action"]["qk99"]]
    stats_p.write_text(json.dumps(raw))

    spec, stats = build_inference_spec(_cfg("robotwin2_clean"), dataset="robotwin2_clean")
    assert spec.per_task_stats
    assert sorted(stats) == ["adjust_bottle", "beat_block_hammer"]
    assert not torch.equal(stats["adjust_bottle"][1]["qk99"], stats["beat_block_hammer"][1]["qk99"])


def test_export_accepts_single_frame_checkpoint(tiny_lerobot_leaf: Callable[..., Path]) -> None:
    """A history_k=1 checkpoint exports with the history cadence: one predict per stride."""
    from reki.inference.spec import build_inference_spec

    tiny_lerobot_leaf("libero")
    single, _ = build_inference_spec(_cfg("libero", history_k=1), benchmark="libero")
    history, _ = build_inference_spec(_cfg("libero"), benchmark="libero")
    assert single == history
    assert single.n_action_steps == round(single.prompt_fps)


def test_dataset_export_rejects_untrained_dataset(tiny_lerobot_leaf: Callable[..., Path]) -> None:
    from reki.inference.spec import build_inference_spec

    tiny_lerobot_leaf("libero")
    tiny_lerobot_leaf("robotwin2_clean")
    with pytest.raises(ValueError, match="not 'robotwin2_clean'"):
        build_inference_spec(_cfg("libero"), dataset="robotwin2_clean")
    with pytest.raises(ValueError, match="exactly one"):
        build_inference_spec(_cfg("libero"))


def test_dataset_export_robot_names(tiny_lerobot_leaf: Callable[..., Path]) -> None:
    """A --dataset export stores the dataset's state and action names for the policy server; a
    --benchmark export has none."""
    from reki.inference.spec import build_inference_spec

    leaf = tiny_lerobot_leaf("libero")
    info_p = leaf / "meta" / "info.json"
    info = json.loads(info_p.read_text())
    state = [f"s{i}.pos" for i in range(info["features"]["observation.state"]["shape"][0])]
    action = [f"a{i}.pos" for i in range(info["features"]["action"]["shape"][0])]
    info["features"]["observation.state"]["names"] = state
    info["features"]["action"]["names"] = action
    info_p.write_text(json.dumps(info))

    cfg = _cfg("libero", max_cameras=3)
    spec, _ = build_inference_spec(cfg, dataset="libero")
    assert spec.state_names == state and spec.state_dim == len(state)
    assert spec.action_names == action
    assert spec.camera_names is not None and len(spec.camera_names) == len(spec.camera_keys)
    bench, _ = build_inference_spec(cfg, benchmark="libero")
    assert bench.state_names is None and bench.action_names is None and bench.camera_names is None
