#!/usr/bin/env python
"""Prepare RoboTwin 2.0 collections for training.

RoboTwin ships as many per-task LeRobot v2.1 repos under one collection root.
Per leaf this script:
  1. Converts v2.1 → v3.0 in place (stock lerobot converter; renames the
     original to ``<leaf>_old``).
  2. Fixes the camera ``shape`` metadata from CHW ``[3,H,W]`` to the repo's
     HWC ``[H,W,3]`` convention, which lerobot expects.
  3. Augments ``qk01..qk99`` quantile stats (the v2.1→v3.0 converter drops
     quantiles; normalization needs them).
Then it writes one ``qk01/qk99`` band, pooled over every leaf, into each leaf: the
collections are absolute-joint, and training rejects per-leaf bands on them. With
``--shard``, rerun once without it after all shards finish (done leaves are skipped).

Idempotent: leaves already at v3.0 are not re-converted; leaves that already
carry ``qk*`` stats are not re-augmented (unless ``--overwrite``).

Usage:
    pixi run --frozen python scripts/data/prepare_robotwin.py                # both collections
    pixi run --frozen python scripts/data/prepare_robotwin.py --collections robotwin2_clean
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from lerobot.datasets.io_utils import load_stats
from lerobot.scripts.convert_dataset_v21_to_v30 import convert_dataset
from shared import (
    apply_global_quantiles,
    has_qk_stats,
    merge_stats,
    parse_shard,
    qk_quantiles,
    read_state_action,
)

from reki.data.robotics.lerobot_io import collection_root, find_collection_leaves, is_collection

logger = logging.getLogger(__name__)


def _leaf_version(leaf: Path) -> str:
    return json.load(open(leaf / "meta" / "info.json")).get("codebase_version", "v0.0")


def _fix_camera_shapes_to_hwc(leaf: Path) -> int:
    """Rewrite CHW ``[3,H,W]`` video shapes to HWC ``[H,W,3]`` in info.json."""
    info_path = leaf / "meta" / "info.json"
    info = json.load(open(info_path))
    fixed = 0
    for feat in info["features"].values():
        if feat.get("dtype") != "video":
            continue
        shape = feat["shape"]
        if len(shape) == 3 and shape[0] == 3:
            feat["shape"] = [shape[1], shape[2], 3]
            fixed += 1
    if fixed:
        json.dump(info, open(info_path, "w"), indent=4)
    return fixed


def _augment_qk_quantiles(leaf: Path, name: str, overwrite: bool) -> bool:
    """Compute qk01..qk99 for state+action and merge into the leaf's stats.json (as
    ``augment_quantiles.py``)."""
    if not overwrite and has_qk_stats(load_stats(leaf)):
        return False
    merge_stats(leaf, qk_quantiles(read_state_action(name, leaf)), only_qk=True)
    return True


def _shard_base_leaves(name: str, shard_i: int, shard_n: int) -> list[Path]:
    """Return this shard's base leaf dirs for a collection.

    The shards split the set of base task names: the dir names without the ``_old`` / ``_v30``
    converter suffix, deduplicated. This set does not change during a conversion, so the shards
    stay disjoint across concurrent array tasks. A live glob changes during a conversion and can
    give the same leaf to two tasks.
    """
    root = collection_root(name)
    base_names = sorted(
        {
            info.parent.parent.name.removesuffix("_old").removesuffix("_v30")
            for info in root.glob("*/meta/info.json")
        }
    )
    if not base_names:
        raise FileNotFoundError(f"No LeRobot leaves for collection '{name}' under {root}")
    leaves = []
    for n in base_names[shard_i::shard_n]:
        leaf = root / n
        if (leaf / "meta" / "info.json").is_file():
            leaves.append(leaf)
        else:
            logger.warning("  skipping %s: no base meta/info.json (orphaned conversion)", n)
    return leaves


def prepare_leaf(leaf: Path, name: str, overwrite: bool) -> None:
    """Convert and repair one RoboTwin leaf for training."""
    if _leaf_version(leaf) < "v3.0":
        logger.info("  converting %s (v2.1 → v3.0)…", leaf.name)
        convert_dataset(
            repo_id=f"local/{name}", root=str(leaf), push_to_hub=False, force_conversion=True
        )
    else:
        logger.info("  %s already v3.0", leaf.name)

    if _fix_camera_shapes_to_hwc(leaf):
        logger.info("  fixed camera shapes → HWC")

    if _augment_qk_quantiles(leaf, name, overwrite):
        logger.info("  wrote qk* quantiles")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collections",
        nargs="+",
        default=["robotwin2_clean", "robotwin2_randomized"],
        help="Registered collections to prepare.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Recompute qk* even if present.")
    parser.add_argument(
        "--shard",
        default=None,
        help="i/N: process only this shard's leaves (stable base-name set, names[i::N]).",
    )
    parser.add_argument("--dry_run", action="store_true", help="list the leaves, write nothing")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    shard_i, shard_n = parse_shard(args.shard)

    for name in args.collections:
        if not is_collection(name):
            raise ValueError(f"{name!r} is not a registered collection")
        leaves = _shard_base_leaves(name, shard_i, shard_n)
        logger.info(
            "Preparing %s: %d leaves (shard %d/%d) under %s",
            name,
            len(leaves),
            shard_i,
            shard_n,
            collection_root(name),
        )
        for leaf in leaves:
            if args.dry_run:
                logger.info("  %s (%s)", leaf.name, _leaf_version(leaf))
                continue
            prepare_leaf(leaf, name, args.overwrite)
        if shard_n == 1 and not args.dry_run:
            logger.info("  writing the global qk01/qk99 of %s", name)
            apply_global_quantiles(find_collection_leaves(name))
    logger.info("Done.")


if __name__ == "__main__":
    main()
