#!/usr/bin/env python3
"""Convert the RoboCasa365 human LeRobot export (v2.1) to a v3.0 collection.

The input is every `pretrain/<category>/<Task>` under the raw root (atomic and composite).
Each output leaf is named `pretrain_<Task>`. The names are unique across categories. The Box
tars have one GR00T-style LeRobot v2.1 export per task (PandaOmron mobile manipulator):

- ``observation.state``: fixed_size_list<f64>[16] (base pose 7 + rel-EE pose 7 + gripper 2)
- ``action``: fixed_size_list<f64>[12] (base_motion 4 + control_mode 1 + eeΔpos 3 + eeΔrot 3
  + gripper 1)
- 3 cameras (robot0_eye_in_hand, agentview_left, agentview_right), 256²
- natural-language instructions in ``episodes.jsonl`` ``tasks`` and ``tasks.jsonl``

The script writes one v3.0 leaf per task with ``shared.convert_groot_leaf`` under the
collection root. The leaves are Human300, the official multitask-track training set.

Then one pass pools all frames of the leaves and writes the same q01/q99, qk01/qk99,
mean/std and min/max into each leaf's stats.json. Global quantiles keep the absolute base
position consistent across kitchens. Constant dims (base_motion and control_mode on the
tabletop atomic subset) have q01 == q99. ``normalization.normalize_feature`` maps them to a
fixed −1, not NaN.

Usage:
    pixi run --frozen python scripts/data/prepare_robocasa365.py [--src SRC --dst DST] [--dry_run]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from shared import apply_global_quantiles, convert_groot_leaf, replace_dir

from reki.paths import datasets_root

logger = logging.getLogger(__name__)

_SRC = Path("robocasa365_atomic_human_raw")
_DST = Path("robocasa365_lerobot_v30_collections") / "robocasa365"
_DIMS = {"observation.state": 16, "action": 12}


def prepare_leaf(src_leaf: Path, dst_leaf: Path, overwrite: bool) -> None:
    """Convert one RoboCasa365 leaf to the v3.0 layout."""
    if not replace_dir(dst_leaf, overwrite):
        logger.info("  %s already converted, skipping", dst_leaf.name)
        return
    logger.info("  converting %s …", dst_leaf.name)
    convert_groot_leaf(src_leaf, dst_leaf, _DIMS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, help=f"default $REKI_DATASETS_ROOT/{_SRC}")
    parser.add_argument("--dst", type=Path, help=f"default $REKI_DATASETS_ROOT/{_DST}")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_global", action="store_true", help="skip global-quantile pass")
    parser.add_argument(
        "--only", default=None, help="substring filter on pretrain/<category>/<Task>"
    )
    parser.add_argument("--dry_run", action="store_true", help="list the leaves, write nothing")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.src = args.src or datasets_root() / _SRC
    args.dst = args.dst or datasets_root() / _DST

    src_leaves = sorted(
        p.parent.parent
        for p in args.src.glob("pretrain/*/*/lerobot/meta/info.json")
        if not args.only or args.only in str(p.parent.parent.relative_to(args.src))
    )
    logger.info("Converting %d RoboCasa365 leaves → %s", len(src_leaves), args.dst)
    for src_leaf in src_leaves:
        dst_leaf = args.dst / f"pretrain_{src_leaf.parent.name}"
        if args.dry_run:
            logger.info("  %s → %s", src_leaf, dst_leaf)
            continue
        prepare_leaf(src_leaf, dst_leaf, args.overwrite)
    if args.dry_run:
        return
    if not args.skip_global:
        apply_global_quantiles(
            sorted(p.parent.parent for p in args.dst.glob("*/meta/info.json")), all_stats=True
        )
    logger.info("Done.")


if __name__ == "__main__":
    main()
