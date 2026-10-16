"""Recompute action/state normalization stats in euler space for the axis-angle datasets.

The stats go to a ``meta/norm_stats_euler.json`` sidecar per leaf. With
``unify_ee_euler_rotation`` set, the loader converts the axis-angle rotvec of these datasets
(``DatasetSpec.action_rot_rep`` / ``state_rot_rep``) to euler, so normalization needs euler
quantiles. The script does not change ``stats.json``, so runs without the flag are not affected.

A dataset whose state roll is shifted by π (``DatasetSpec.recenter_state_roll``) gets
``meta/norm_stats_euler_roll.json`` instead, with the shifted roll.

Rotation blocks (``transforms.rot_blocks``) are in post-slice coords. The sidecar is in
pre-slice coords and the loader slices it, so ``shared.raw_rot_blocks`` maps the blocks back
through the dataset's slices. The conversion is per feature: a dataset can have an axis-angle
action and an euler state.

Run:
    pixi run --frozen python scripts/data/recompute_euler_stats.py [--dry_run]
    pixi run --frozen python scripts/data/recompute_euler_stats.py --datasets libero
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from lerobot.datasets.compute_stats import DEFAULT_QUANTILES, get_feature_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from shared import alias_qk_band, axis_angle_to_euler, raw_rot_blocks

from reki.data.robotics.lerobot_io import is_collection, resolve_lerobot_dirs
from reki.data.robotics.lerobot_patches import install_lerobot_patches
from reki.data.robotics.registry import AXIS_ANGLE, DATASETS
from reki.data.robotics.transforms import recenter_euler_roll, rot_blocks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_FEATURES = (
    ("action", {n for n, s in DATASETS.items() if s.action_rot_rep == AXIS_ANGLE}),
    ("observation.state", {n for n, s in DATASETS.items() if s.state_rot_rep == AXIS_ANGLE}),
)


def _leaf_datasets(name: str) -> list[tuple[str, LeRobotDataset]]:
    """Resolve a dataset to (label, LeRobotDataset) pairs, as training does
    (``resolve_lerobot_dirs`` under ``$REKI_DATASETS_ROOT``).

    A multi-repo collection gives one LeRobotDataset per leaf. Each leaf has its own stats and
    its own euler sidecar, as in the per-leaf normalization of CombinedLeRobotDataset.
    """
    install_lerobot_patches()
    return [
        (
            f"{name}/{leaf.name}" if is_collection(name) else name,
            LeRobotDataset(repo_id=f"local/{leaf.name}", root=leaf, revision="local"),
        )
        for leaf in resolve_lerobot_dirs(name)
    ]


def _stats_for(name: str, label: str, ds: LeRobotDataset, overwrite: bool, dry_run: bool) -> bool:
    """Write the euler sidecar for one leaf. ``name`` is the registry member key (selects
    embodiment rot-blocks + per-feature axis-angle membership); ``label`` is for logging."""
    root = Path(ds.root)
    spec = DATASETS[name]
    recenter = spec.recenter_state_roll
    out_path = (
        root / "meta" / ("norm_stats_euler_roll.json" if recenter else "norm_stats_euler.json")
    )
    if out_path.exists() and not overwrite:
        logger.info("%s: %s exists, skipping (use --overwrite).", label, out_path.name)
        return False
    logger.info("%s: loading from %s", label, root)
    blocks = rot_blocks(spec.embodiment)
    _raw_slices = {"action": spec.action_slice, "observation.state": spec.state_slice}

    sidecar: dict[str, dict] = {}
    for key, rep_set in _FEATURES:
        if key not in ds.features:
            continue
        data = np.asarray(ds.select_columns([key])[key], dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]
        if name in rep_set:
            data = axis_angle_to_euler(data, raw_rot_blocks(blocks, _raw_slices[key]))
            if recenter and key == "observation.state":
                recenter_euler_roll(data, raw_rot_blocks(blocks, _raw_slices[key]))
        s = get_feature_stats(data, axis=0, keepdims=False, quantile_list=DEFAULT_QUANTILES)
        s = {k: np.asarray(v).reshape(-1).tolist() for k, v in s.items()}
        alias_qk_band(s)
        sidecar[key] = s
        rot_dims = [d for lo, hi in raw_rot_blocks(blocks, _raw_slices[key]) for d in range(lo, hi)]
        logger.info(
            "  %s: %d dims%s, rot q01=%s q99=%s",
            key,
            len(s["min"]),
            "" if name in rep_set else " (euler-already, unconverted)",
            [round(s["q01"][d], 3) for d in rot_dims],
            [round(s["q99"][d], 3) for d in rot_dims],
        )

    if dry_run:
        logger.info("%s: would write %s", label, out_path)
        return False
    out_path.write_text(json.dumps(sidecar))
    logger.info("%s: wrote %s", label, out_path)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Recompute euler-space norm stats for axis-angle single_arm_ee_euler members."
    )
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="compute and report, write nothing")
    args = ap.parse_args()
    named = bool(args.datasets)
    if not named:
        args.datasets = sorted(_FEATURES[0][1])

    wrote = 0
    missing: list[str] = []
    for name in args.datasets:
        try:
            leaves = _leaf_datasets(name)
        except FileNotFoundError as e:
            logger.warning("%s: not found on disk, skipping (%s)", name, e)
            missing.append(name)
            continue
        for label, ds in leaves:
            if _stats_for(name, label, ds, args.overwrite, args.dry_run):
                wrote += 1
    logger.info("Done. Wrote euler sidecars for %d dataset(s)/leaf/leaves.", wrote)
    if named and missing:
        raise SystemExit(f"Not found under $REKI_DATASETS_ROOT: {missing}")


if __name__ == "__main__":
    main()
