"""Write a train/val episode split for a dataset that is its own val set.

The script holds out `--per_task` episodes of each task and writes them to
`meta/dropped_episodes.json`. The RatioSampler (kept_local_frames) does not draw these episodes
for training. The val loader reads the list in `registry.VAL_DATASETS[<dataset>]`, so the two
lists must agree. Run this script only when you also update that list.

The val picks are evenly spaced in each task's episode block. Episodes are contiguous and in
recording order, so the tail would hold out one end of any drift during collection.

`--cap` also trims each task to a fixed episode count (e.g. a protocol of 50 demos per task on
a dataset that has 100 for some tasks). The capped-out episodes also go to
`dropped_episodes.json`, so training does not use them. The file lists all episodes that
training does not use. Only `VAL_DATASETS` says which of them are val. The kept episodes are
evenly spaced, for the same reason as the val picks.

Usage:
    pixi run --frozen python scripts/data/make_val_split.py --dataset <dataset> --per_task 2 --dry_run
    pixi run --frozen python scripts/data/make_val_split.py --dataset <dataset> --per_task 5 --cap 50
"""

import argparse
import json

import pandas as pd

from reki.data.robotics.lerobot_io import dataset_root, find_lerobot_dir


def make_val_split(dataset: str, per_task: int, cap: int | None, dry_run: bool) -> None:
    root = find_lerobot_dir(dataset, root=dataset_root(dataset))
    shards = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no episode metadata under {root / 'meta' / 'episodes'}")
    episodes = pd.concat([pd.read_parquet(f) for f in shards], ignore_index=True)
    episodes["task"] = episodes["tasks"].apply(lambda t: t[0])

    held_out: list[int] = []
    val_only: list[int] = []
    for task, grp in episodes.groupby("task", sort=True):
        all_idx = sorted(int(e) for e in grp["episode_index"])
        if cap is not None and cap < len(all_idx):
            keep = [all_idx[len(all_idx) * i // cap] for i in range(cap)]
            dropped_by_cap = [e for e in all_idx if e not in set(keep)]
            held_out.extend(dropped_by_cap)
            idx = keep
            print(
                f"{task[:58]:60s} {len(all_idx):3d} eps -> cap {cap} (excluded {len(dropped_by_cap)})"
            )
        else:
            idx = all_idx
        picks = [idx[len(idx) * (i + 1) // (per_task + 1)] for i in range(per_task)]
        val_only.extend(picks)
        held_out.extend(picks)
        print(f"{task[:58]:60s} {len(idx):3d} eps -> val {picks}")

    held_out = sorted(held_out)
    val_only = sorted(val_only)
    n_val_frames = int(
        episodes[episodes["episode_index"].isin(val_only)]["length"].sum()  # type: ignore[arg-type]
    )
    total = int(episodes["length"].sum())
    print(
        f"\nval {len(val_only)}/{len(episodes)} episodes, "
        f"{n_val_frames:,}/{total:,} frames ({100 * n_val_frames / total:.1f}%); "
        f"not trained on: {len(held_out)} episodes"
    )
    print(f"\nVAL_DATASETS[{dataset!r}] = {val_only}")
    if cap is not None:
        print(f"dropped_episodes.json (val + capped-out) = {len(held_out)} episodes")

    out = root / "meta" / "dropped_episodes.json"
    if dry_run:
        print(f"\n(dry run; would write {out})")
        return
    out.write_text(json.dumps(held_out))
    print(f"\nwrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", required=True, help="registered dataset name")
    parser.add_argument("--per_task", type=int, default=2, help="val episodes per task")
    parser.add_argument("--cap", type=int, default=None, help="train episodes per task")
    parser.add_argument("--dry_run", action="store_true", help="print the split, write nothing")
    args = parser.parse_args()
    make_val_split(args.dataset, args.per_task, args.cap, args.dry_run)


if __name__ == "__main__":
    main()
