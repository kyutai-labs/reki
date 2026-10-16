#!/usr/bin/env python3
"""Build "pseudo-v3.0" LeRobot datasets: the per-episode parquet/video layout is kept, and
``meta/`` is rebuilt to the v3.0 schema. The script does not use the lerobot converter, which
runs out of memory on large datasets and fails on ``-c copy`` videos. It converts one v2.x
root to v3.0; ``--dry_run`` lists what it would write.

Usage:
    pixi run --frozen python scripts/data/build_v30.py --input SRC --output DST [--link_data]
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from shared import V30_META_FEATURES, V30_VIDEO_PATH, write_v30_episodes, write_v30_tasks

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def convert(V21_ROOT: Path, V30_ROOT: Path, link_data: bool = False) -> None:
    """Build a pseudo-v3.0 dataset from a single v2.x LeRobot root.

    v2.1 inputs have a per-episode ``episodes_stats.jsonl``, which is aggregated here. v2.0
    inputs have one aggregated ``meta/stats.json``, often with q01/q99. It is copied as is,
    because aggregation would drop the quantiles. The video ``to_timestamp`` comes from the last
    parquet timestamp of each episode, not from a global fps, so mixed-fps datasets are correct.
    """
    V30_ROOT.mkdir(parents=True, exist_ok=True)

    logger.info("Building info.json...")
    info = json.load(open(V21_ROOT / "meta" / "info.json"))
    info["codebase_version"] = "v3.0"

    for k, v in V30_META_FEATURES.items():
        if k not in info["features"]:
            info["features"] = {k: v, **info["features"]}

    (V30_ROOT / "meta").mkdir(parents=True, exist_ok=True)
    with open(V30_ROOT / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    fps = info.get("fps", 30)

    logger.info("%s data files...", "Linking" if link_data else "Copying")
    src_files = sorted(glob.glob(str(V21_ROOT / "data" / "**" / "*.parquet"), recursive=True))
    for i, src in enumerate(src_files):
        rel = os.path.relpath(src, V21_ROOT)
        dst = V30_ROOT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        if link_data:
            os.symlink(src, dst)
        else:
            shutil.copy2(src, dst)
        if (i + 1) % 1000 == 0:
            logger.info("  %d/%d data files", i + 1, len(src_files))
    logger.info("Processed %d data files", len(src_files))

    logger.info("Building tasks.parquet...")
    write_v30_tasks(V21_ROOT, V30_ROOT)

    logger.info("Building episodes.parquet...")
    chunks_size = info.get("chunks_size", 1000)

    def _video_end(idx: int, length: int) -> float:
        pq_path = (
            V21_ROOT / "data" / f"chunk-{idx // chunks_size:03d}" / f"episode_{idx:06d}.parquet"
        )
        if pq_path.exists() and length > 1:
            ts_col = pq.read_table(pq_path, columns=["timestamp"]).column("timestamp")
            max_ts = float(ts_col[len(ts_col) - 1].as_py())
            return max_ts + max_ts / (length - 1)
        return length / fps

    n_episodes = write_v30_episodes(V21_ROOT, V30_ROOT, info, _video_end)
    logger.info("Wrote %d episodes", n_episodes)

    ep_stats_path = V21_ROOT / "meta" / "episodes_stats.jsonl"
    if ep_stats_path.exists():
        logger.info("Building stats from episodes_stats.jsonl...")
        from lerobot.datasets.io_utils import write_stats

        ep_stats_list = [json.loads(line)["stats"] for line in open(ep_stats_path)]
        all_keys = set()
        for es in ep_stats_list:
            all_keys.update(es.keys())

        aggregated = {}
        for key in sorted(all_keys):
            vals = [es[key] for es in ep_stats_list if key in es]
            try:
                means = np.stack([np.array(v["mean"]) for v in vals])
                counts = np.stack([np.array(v["count"]) for v in vals])
                mins = np.stack([np.array(v["min"]) for v in vals])
                maxs = np.stack([np.array(v["max"]) for v in vals])
                stds = np.stack([np.array(v["std"]) for v in vals])
                # Per-episode counts, broadcast over the feature dims of mean/std.
                w = counts.reshape(len(vals), *[1] * (means.ndim - 1))
                total_count = counts.sum(axis=0)
                mean = np.sum(means * w, axis=0) / w.sum(axis=0)
                variance = np.sum(w * (stds**2 + (means - mean) ** 2), axis=0) / w.sum(axis=0)
                aggregated[key] = {
                    "min": np.min(mins, axis=0),
                    "max": np.max(maxs, axis=0),
                    "mean": mean,
                    "std": np.sqrt(variance),
                    "count": total_count,
                }
            except Exception:
                continue
        write_stats(aggregated, V30_ROOT)
        logger.info("Wrote stats for %d features", len(aggregated))
    else:
        logger.info("No episodes_stats.jsonl; copying aggregated stats.json verbatim")
        shutil.copy2(V21_ROOT / "meta" / "stats.json", V30_ROOT / "meta" / "stats.json")

    videos_link = V30_ROOT / "videos"
    if videos_link.exists():
        videos_link.unlink()
    os.symlink(V21_ROOT / "videos", videos_link)
    logger.info("Symlinked videos")

    info["video_path"] = V30_VIDEO_PATH
    with open(V30_ROOT / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    logger.info("Done! v3.0 dataset at %s", V30_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True, help="v2.x dataset root")
    parser.add_argument("--output", type=Path, required=True, help="v3.0 output root")
    parser.add_argument(
        "--link_data",
        action="store_true",
        help="symlink data parquets instead of copying (schema unchanged; saves disk)",
    )
    parser.add_argument("--dry_run", action="store_true", help="list what would be written")

    args = parser.parse_args()
    if args.dry_run:
        logger.info("would convert %s → %s", args.input, args.output)
        return
    convert(args.input, args.output, args.link_data)


if __name__ == "__main__":
    main()
