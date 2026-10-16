"""Measure which stored EE-position axis is left-right in LIBERO.

Do not assume robosuite's world-frame convention. A wrong axis gives no error: the mirror stays
self-consistent but reflects the world about the wrong plane.

The measurement uses a task label. LIBERO task 0 is the only task that names both sides: "put
the white mug on the left plate and put the yellow and white mug on the right plate". Its
episodes must move along the lateral axis, about symmetrically around the robot's sagittal
plane, and stay compact on the other two axes. The other tasks are controls.

``--write_fixture`` writes the per-episode position extrema, so
``tests/data/test_mirror_libero.py`` derives the axis from committed data. ``--report`` prints
the table.

    pixi run --frozen python scripts/data/derive_libero_mirror_axis.py --report
    pixi run --frozen python scripts/data/derive_libero_mirror_axis.py --write_fixture
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np

LEFT_RIGHT_TASK = (
    "put the white mug on the left plate and put the yellow and white mug on the right plate"
)
CROSSING_M = 0.05
FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "libero_ee_extrema.json"


def load_episode_extrema(root: Path) -> dict[str, Any]:
    """Per-episode min/max of the 3 stored EE-position dims, grouped by task string."""
    import pandas as pd

    files = sorted(glob.glob(str(root / "meta" / "episodes" / "chunk-*" / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no episode metadata under {root}")
    ep = pd.concat([pd.read_parquet(f) for f in files])
    lo = np.stack(ep["stats/observation.state/min"].to_numpy())[:, :3]
    hi = np.stack(ep["stats/observation.state/max"].to_numpy())[:, :3]
    tasks = [t[0] for t in ep["tasks"]]
    return {
        "task_of_episode": tasks,
        "pos_min": np.round(lo, 4).tolist(),
        "pos_max": np.round(hi, 4).tolist(),
    }


def lateral_axis(fixture: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """The measured lateral position axis and the evidence for it.

    Two statistics must agree. Per-task mean excursion: the left/right task must rank first on
    the lateral axis. Zero-crossing rate in that task's episodes: the episodes must cross zero on
    the lateral axis only.
    """
    tasks = np.asarray(fixture["task_of_episode"])
    lo = np.asarray(fixture["pos_min"], dtype=float)
    hi = np.asarray(fixture["pos_max"], dtype=float)
    if LEFT_RIGHT_TASK not in set(tasks):
        raise ValueError(f"fixture is missing the discriminating task {LEFT_RIGHT_TASK!r}")
    rng = hi - lo
    names = sorted(set(tasks))
    per_task = np.stack([rng[tasks == t].mean(axis=0) for t in names])
    lr_row = per_task[names.index(LEFT_RIGHT_TASK)]

    is_lr = tasks == LEFT_RIGHT_TASK
    crossing = ((lo[is_lr] < -CROSSING_M) & (hi[is_lr] > CROSSING_M)).mean(axis=0)
    ranks = [int((per_task[:, i] >= lr_row[i]).sum()) for i in range(3)]

    axis = int(np.argmax(crossing))
    if ranks[axis] != 1 or crossing[axis] < 0.9:
        raise ValueError(
            f"axis {axis} is not decisive: excursion rank {ranks[axis]}/{len(names)}, "
            f"crossing rate {crossing[axis]:.2f} — re-measure before trusting a sign"
        )
    runner_up = sorted(crossing)[-2]
    if runner_up > 0.5:
        raise ValueError(f"two axes cross comparably ({crossing}); the measurement is ambiguous")
    return axis, {
        "n_tasks": len(names),
        "n_lr_episodes": int(is_lr.sum()),
        "lr_excursion_m": lr_row.tolist(),
        "other_excursion_m": np.delete(per_task, names.index(LEFT_RIGHT_TASK), 0).mean(0).tolist(),
        "excursion_rank": ranks,
        "crossing_rate": crossing.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", action="store_true", help="print the measurement table")
    ap.add_argument("--write_fixture", action="store_true", help=f"(re)write {FIXTURE.name}")
    args = ap.parse_args()

    from reki.data.robotics.lerobot_io import dataset_root

    if args.write_fixture or args.report:
        fixture = load_episode_extrema(dataset_root("libero") / "libero")
    else:
        fixture = json.loads(FIXTURE.read_text())

    axis, ev = lateral_axis(fixture)
    print(f"measured lateral position axis: {axis}")
    if args.report:
        for i in range(3):
            print(
                f"  dim{i}: left/right task {ev['lr_excursion_m'][i]:.3f} m | "
                f"others {ev['other_excursion_m'][i]:.3f} m | "
                f"rank {ev['excursion_rank'][i]}/{ev['n_tasks']} | "
                f"crosses in {ev['crossing_rate'][i] * 100:.0f}% of "
                f"{ev['n_lr_episodes']} left/right episodes"
            )
    if args.write_fixture:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(fixture, separators=(",", ":")))
        print(f"wrote {FIXTURE} ({FIXTURE.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
