#!/usr/bin/env python
"""Write ``qk01/qk10/qk50/qk90/qk99`` action / state quantiles into ``meta/stats.json``.

The ``qk*`` keys are written next to any upstream ``q01/q99``. The loader requires
``qk01/qk99``. Both modes read straight from the data parquets:

- ``--leaves DIR``: every v3.0 leaf under DIR. ``--shard i/N`` takes ``leaves[i::N]``.
  ``--global`` instead writes one ``qk01/qk99`` band, pooled over the frames of every leaf,
  into each leaf: an absolute-control collection needs it.
- ``--leaf DIR``: one leaf.

Per-leaf quantiles come from lerobot's ``get_feature_stats`` and only ``qk*`` is merged in.
Leaves that already have ``qk*`` are skipped unless ``--overwrite`` (``--global`` always
writes). Failures are collected and raised at the end. ``--dry_run`` lists the items that
would be written.

    pixi run --frozen python scripts/data/augment_quantiles.py --leaves <dir> --shard 0/16
    pixi run --frozen python scripts/data/augment_quantiles.py --leaves <collection root> --global
    pixi run --frozen python scripts/data/augment_quantiles.py --leaf <leaf dir>
"""

from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from lerobot.datasets.io_utils import load_stats
from shared import (
    STATE_ACTION,
    apply_global_quantiles,
    has_qk_stats,
    merge_stats,
    parse_shard,
    qk_quantiles,
)

logger = logging.getLogger(__name__)


def _read_parquets(leaf: Path) -> dict[str, np.ndarray]:
    files = sorted(glob.glob(str(leaf / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"{leaf}: no data parquets")
    parts: dict[str, list[np.ndarray]] = {col: [] for col in STATE_ACTION}
    for f in files:
        names = pq.read_schema(f).names
        table = pq.read_table(f, columns=[c for c in STATE_ACTION if c in names])
        for col in table.column_names:
            arr = table.column(col).combine_chunks()
            values = arr.flatten().to_numpy(zero_copy_only=False).astype(np.float32)
            parts[col].append(values.reshape(len(arr), -1))
    arrays = {col: np.concatenate(p, axis=0) for col, p in parts.items() if p}
    if not arrays:
        raise ValueError(f"{leaf}: no action/state columns")
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--leaves", type=Path, help="dir of v3.0 leaves (parquet reader)")
    which.add_argument("--leaf", type=Path, help="one v3.0 leaf (parquet reader)")
    parser.add_argument("--shard", help="i/N: leaves[i::N] (with --leaves)")
    parser.add_argument(
        "--global",
        dest="global_band",
        action="store_true",
        help="with --leaves: write one qk01/qk99 band, pooled over all leaves, into each",
    )
    parser.add_argument("--overwrite", action="store_true", help="recompute existing qk*")
    parser.add_argument("--dry_run", action="store_true", help="list the items, write nothing")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.shard and args.leaves is None:
        parser.error("--shard needs --leaves")
    if args.global_band and (args.leaves is None or args.shard):
        parser.error("--global needs --leaves and no --shard")

    if args.leaves:
        shard_i, shard_n = parse_shard(args.shard)
        leaves = sorted(p for p in args.leaves.iterdir() if (p / "meta" / "info.json").exists())[
            shard_i::shard_n
        ]
    else:
        leaves = [args.leaf]
    logger.info("%d item(s)", len(leaves))

    if args.global_band:
        if args.dry_run:
            logger.info("would write global qk01/qk99 to %d leaves", len(leaves))
            return
        apply_global_quantiles(leaves)
        logger.info("global qk01/qk99 written to %d leaves", len(leaves))
        return

    done = skipped = 0
    failed: list[str] = []
    for leaf in leaves:
        if not args.overwrite and has_qk_stats(load_stats(leaf)):
            skipped += 1
            continue
        if args.dry_run:
            logger.info("would write qk*: %s", leaf)
            continue
        try:
            merge_stats(leaf, qk_quantiles(_read_parquets(leaf)), only_qk=True)
        except Exception:
            logger.exception("FAILED %s", leaf.name)
            failed.append(leaf.name)
            continue
        done += 1
        logger.info("qk* written: %s", leaf.name)
    logger.info("%d augmented, %d skipped, %d failed", done, skipped, len(failed))
    if failed:
        raise RuntimeError(f"{len(failed)} failed: {failed}")


if __name__ == "__main__":
    main()
