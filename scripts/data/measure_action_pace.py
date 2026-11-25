#!/usr/bin/env python
"""Measure the egocentric action pace (``TrainConfig.action_pace``).

Human hands move faster than robot arms. ``action_pace`` < 1 slows an ego dataset's action
chunk at load time (``transforms.interpolate_wrist_chunk``) so its wrist speed matches the robot
data. This script measures the factor:

1. Speed: the net translation speed of the faster end effector over non-overlapping 0.5 s
   windows (m/s), from the absolute EE positions of each episode. Ego: the stored world wrists
   (action ``0:3`` and ``16:19``); a window with an untracked (all-zero) hand row at either end
   leaves that hand out. Robot reference: datasets with metric EE positions in
   ``observation.state``. Each dataset gives the median over its moving windows (> 5 mm/s).
2. Robot reference speed: the median of the robot dataset medians.
3. Pace: the factor whose paced ego trajectory (lerp at fractional frame ``i * pace``, as in
   training) has the reference median speed, found by bisection; 1.0 when the ego data is
   already at most as fast as the reference.

Each dataset is read from its first parquet files up to ``--max_frames`` frames.

    pixi run python scripts/data/measure_action_pace.py [--ego egodex xperience]
        [--robot libero interndata_franka] [--robot_ref M_PER_S] [--max_frames N]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from reki.data.robotics.lerobot_io import resolve_lerobot_dirs

EGO_SPEC = ("action", [slice(0, 3), slice(16, 19)])
ROBOT_SPECS: dict[str, tuple[str, list[slice]]] = {
    "libero": ("observation.state", [slice(0, 3)]),
    "interndata_franka": ("observation.state", [slice(0, 3)]),
}
HAND_BLOCK = 16
MOVING_M_S = 0.005
WINDOW_S = 0.5


def iter_episodes(
    leaves: list[Path], column: str, max_frames: int
) -> Iterator[tuple[np.ndarray, float]]:
    """``(positions [T, D], fps)`` per episode, from the first parquet files up to ``max_frames``."""
    n = 0
    for leaf in leaves:
        fps = float(json.loads((leaf / "meta" / "info.json").read_text())["fps"])
        for f in sorted((leaf / "data").rglob("*.parquet")):
            table = pq.read_table(f, columns=[column, "episode_index"])
            vec = np.stack(table[column].to_numpy(zero_copy_only=False)).astype(np.float64)
            eps = table["episode_index"].to_numpy()
            for e in np.unique(eps):
                pos = vec[eps == e]
                yield pos, fps
                n += len(pos)
            if n >= max_frames:
                return


def tracked_rows(vec: np.ndarray, ee: list[slice], ego: bool) -> list[np.ndarray]:
    """``[T]`` bool per end effector: the rows where it is tracked (ego hand block not all zero)."""
    if not ego:
        return [np.ones(len(vec), bool) for _ in ee]
    return [np.any(vec[:, s.start : s.start + HAND_BLOCK] != 0, axis=1) for s in ee]


def window_speeds(
    pos: np.ndarray, valid: list[np.ndarray], ee: list[slice], fps: float
) -> np.ndarray:
    """Net speed (m/s) of the faster tracked end effector over non-overlapping 0.5 s windows."""
    w = max(1, round(WINDOW_S * fps))
    if len(pos) <= w:
        return np.zeros(0)
    end = np.arange(w, len(pos), w)
    start = end - w
    per_ee = []
    for s, ok in zip(ee, valid):
        speed = np.linalg.norm(pos[end, s] - pos[start, s], axis=1) * fps / w
        per_ee.append(np.where(ok[start] & ok[end], speed, np.nan))
    per_ee = np.stack(per_ee)
    keep = ~np.isnan(per_ee).all(axis=0)
    return np.nanmax(per_ee[:, keep], axis=0)


def paced(
    vec: np.ndarray, valid: list[np.ndarray], pace: float
) -> tuple[np.ndarray, list[np.ndarray]]:
    """The trajectory at fractional frames ``i * pace`` (lerp); a row next to an untracked one is
    untracked."""
    f = np.arange(int(len(vec) / pace)) * pace
    i0 = np.floor(f).astype(int)
    i1 = np.minimum(i0 + 1, len(vec) - 1)
    a = (f - i0)[:, None]
    return (1 - a) * vec[i0] + a * vec[i1], [ok[i0] & ok[i1] for ok in valid]


def speeds(
    leaves: list[Path], column: str, ee: list[slice], ego: bool, max_frames: int, pace: float = 1.0
) -> np.ndarray:
    """Window speeds over up to ``max_frames`` source frames, at action pace ``pace``."""
    out = []
    for vec, fps in iter_episodes(leaves, column, max_frames):
        valid = tracked_rows(vec, ee, ego)
        if pace != 1.0:
            vec, valid = paced(vec, valid, pace)
        out.append(window_speeds(vec, valid, ee, fps))
    if not out:
        raise RuntimeError(f"no episodes under {leaves[0].parent}")
    return np.concatenate(out)


def moving_median(v: np.ndarray) -> float:
    return float(np.median(v[v > MOVING_M_S]))


def solve_pace(leaves: list[Path], ref: float, max_frames: int, tol: float = 0.005) -> float:
    """The pace in (0, 1] whose paced ego moving median equals ``ref`` (bisection)."""
    column, ee = EGO_SPEC
    if moving_median(speeds(leaves, column, ee, True, max_frames)) <= ref:
        return 1.0
    lo, hi = 0.05, 1.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if moving_median(speeds(leaves, column, ee, True, max_frames, mid)) > ref:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ego", nargs="+", default=["egodex", "xperience"])
    ap.add_argument("--robot", nargs="+", default=list(ROBOT_SPECS), choices=list(ROBOT_SPECS))
    ap.add_argument(
        "--robot_ref", type=float, help="reference speed in m/s (skips the robot datasets)"
    )
    ap.add_argument(
        "--max_frames", type=int, default=100_000, help="source frames read per dataset"
    )
    args = ap.parse_args()

    print(
        f"{'dataset':20s} {'windows':>8s} {'moving':>7s} {'median':>7s}   [m/s, {WINDOW_S} s net]"
    )

    def report(name: str, v: np.ndarray) -> float:
        med = moving_median(v)
        print(f"{name:20s} {len(v):>8d} {np.mean(v > MOVING_M_S):>7.1%} {med:>7.4f}")
        return med

    ref = args.robot_ref
    if ref is None:
        medians = [
            report(
                name, speeds(resolve_lerobot_dirs(name), *ROBOT_SPECS[name], False, args.max_frames)
            )
            for name in args.robot
        ]
        ref = float(np.median(medians))
    print(f"robot reference: {ref:.4f} m/s")
    for name in args.ego:
        leaves = resolve_lerobot_dirs(name)
        report(name, speeds(leaves, *EGO_SPEC, True, args.max_frames))
        pace = solve_pace(leaves, ref, args.max_frames)
        emitted = moving_median(speeds(leaves, *EGO_SPEC, True, args.max_frames, pace))
        print(f"  {name}: action_pace {pace:.2f} (paced median {emitted:.4f} m/s)")


if __name__ == "__main__":
    main()
