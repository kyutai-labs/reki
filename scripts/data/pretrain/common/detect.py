"""Episode detectors whose hits go to ``meta/dropped_episodes.json``.

``frozen``: the robot state never moves. Per episode and raw state dim, the std over the
episode's frames; a dim is frozen in an episode when its std < ``thr``. By-design-constant dims
(frozen in >= ``const_frac`` of episodes and never moving in any) are excluded; an episode is
dropped when more than ``drop_frac`` of the other dims are frozen.

``sudden_change`` (episode discard): on the trained signal (registry
slices, axis-angle → euler), per dim a residual to a median+Savitzky–Golay trend and the 2nd/3rd
finite differences. Per leaf, each threshold is ``max(scale × pooled P95, floor)``. A frame is
flagged when residual > thr and (accel > thr or jerk > thr); an episode with a flagged frame is
dropped. Gripper dims (``gripper`` in the feature name) are skipped.

``ur_frozen_arm`` (RoboMIND UR): the recorded follower arm does not move. An episode is dropped
when the max over the 6 arm joints (state dims 0-5) of the per-episode std is below
``UR_ARM_THR`` rad; the gripper (dim 6) moves in these episodes too. The arm-motion
distribution is bimodal (~0 or ~0.3 rad).

``failure`` (DROID): the collector labeled the episode as not successful (upstream
``is_episode_successful`` False).

``short_video``: an episode whose video, on any camera, holds fewer frames than its data rows
(an upstream mp4 cut short; the loader cannot read the last rows' frames).

``video_cuts`` (DROID): video files that stitch several recordings. Per video file of the
``top`` camera, ffmpeg scene detection (160×90, scene score > ``CUT_SCENE``); an episode has an
internal cut when one falls more than ``CUT_EDGE_S`` inside its span. A file where at least
``CUT_FILE_FRAC`` of the episodes have one is dropped whole (each of its episodes).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.signal import medfilt, savgol_filter

from .v30 import data_files, read_columns

FROZEN_THR = 0.01
FROZEN_CONST_FRAC = 0.95
FROZEN_DROP_FRAC = 0.70

SUDDEN_SCALE = 4.0
SUDDEN_FLOOR = 0.05
SMOOTH_WINDOW = 15
MED_KERNEL = 5
POLYORDER = 2
PERCENTILE = 95.0
EPS = 1e-8

CUT_SCENE = 0.5
CUT_EDGE_S = 0.5
CUT_FILE_FRAC = 0.25

UR_ARM_DIMS = slice(0, 6)
UR_ARM_THR = 0.02

CONTEXT_SKIP_RAW: dict[str, dict[str, set[int]]] = {"galaxea": {"state": {32, 33, 34, 35}}}
"""Raw dims ``sudden_change`` skips beyond the grippers."""


def state_std(leaf: Path) -> tuple[np.ndarray, list[int]]:
    """Per-episode per-dim std of the raw state ``[n_eps, D]`` and the sorted episode indices.

    Sums and squared sums accumulate per episode in float64 over the data files, so episodes
    spanning files are handled.
    """
    count: dict[int, int] = {}
    s1: dict[int, np.ndarray] = {}
    s2: dict[int, np.ndarray] = {}
    for f in data_files(leaf):
        cols = read_columns(f, ["observation.state", "episode_index"])
        epi = cols["episode_index"]
        state = cols["observation.state"].astype(np.float64)
        for ep in np.unique(epi):
            m = epi == ep
            ep = int(ep)
            seg = state[m]
            count[ep] = count.get(ep, 0) + int(m.sum())
            s1[ep] = s1.get(ep, 0.0) + seg.sum(0)
            s2[ep] = s2.get(ep, 0.0) + (seg**2).sum(0)
    eps = sorted(count)
    rows = [np.sqrt(np.maximum(s2[ep] / count[ep] - (s1[ep] / count[ep]) ** 2, 0.0)) for ep in eps]
    return np.stack(rows), eps


def frozen_episodes(
    std: np.ndarray,
    ep_ids: list[int],
    thr: float = FROZEN_THR,
    const_frac: float = FROZEN_CONST_FRAC,
    drop_frac: float = FROZEN_DROP_FRAC,
) -> list[int]:
    """The episodes of a ``[n_eps, D]`` std matrix whose live dims are mostly frozen."""
    frozen = std < thr
    const_dim = (frozen.mean(0) >= const_frac) & (std.max(0) < thr)
    live = np.flatnonzero(~const_dim)
    if len(live) == 0:
        return []
    drop = frozen[:, live].mean(1) > drop_frac
    return [ep_ids[i] for i in np.flatnonzero(drop)]


def frozen(leaf: Path, name: str) -> list[int]:
    """The frozen episodes of one leaf."""
    return frozen_episodes(*state_std(leaf))


def _odd_clamp(win: int, n: int) -> int:
    win = min(win, n)
    if win % 2 == 0:
        win -= 1
    return max(win, 1)


def _smooth(x: np.ndarray, savgol_win: int) -> np.ndarray:
    """Median filter twice, then Savitzky–Golay when the episode is long enough."""
    mk = _odd_clamp(MED_KERNEL, len(x))
    y = medfilt(medfilt(x, mk), mk)
    sw = _odd_clamp(savgol_win, len(x))
    if sw > POLYORDER:
        y = savgol_filter(y, sw, POLYORDER)
    return y


def deviations(x: np.ndarray, savgol_win: int = SMOOTH_WINDOW) -> tuple[np.ndarray, ...]:
    """(residual, accel, jerk) of one dim on its length-N frame axis; the finite differences
    sit on interior frames, zero at the ends."""
    n = len(x)
    resid = np.abs(x - _smooth(x, savgol_win))
    acc = np.zeros(n)
    jerk = np.zeros(n)
    if n >= 3:
        acc[1 : n - 1] = np.abs(np.diff(x, n=2))
    if n >= 4:
        jerk[1 : n - 2] = np.abs(np.diff(x, n=3))
    return resid, acc, jerk


def _pooled(chunks: list[np.ndarray]) -> float:
    vals = [c for c in chunks if c.size]
    return float(np.percentile(np.concatenate(vals), PERCENTILE)) if vals else 0.0


def sudden_change_episodes(
    episodes: list[tuple[int, dict[str, np.ndarray]]],
    skip: dict[str, set[int]],
    scale: float = SUDDEN_SCALE,
    floor: float = SUDDEN_FLOOR,
) -> list[int]:
    """The episodes with a flagged frame, from ``(index, {"state": [N, Ds], "action": [N, Da]})``
    of one leaf; thresholds are pooled over these episodes."""
    devs: list[tuple[int, int, dict[tuple[str, int], tuple[np.ndarray, ...]]]] = []
    pools: dict[tuple[str, int], tuple[list, list, list]] = {}
    for index, arrays in episodes:
        dev: dict[tuple[str, int], tuple[np.ndarray, ...]] = {}
        length = 0
        for source, arr in arrays.items():
            length = len(arr)
            for dim in range(arr.shape[1]):
                if dim in skip.get(source, set()):
                    continue
                resid, acc, jerk = deviations(arr[:, dim].astype(np.float64))
                dev[(source, dim)] = (resid, acc, jerk)
                rl, al, jl = pools.setdefault((source, dim), ([], [], []))
                rl.append(resid)
                al.append(acc[1 : length - 1])
                jl.append(jerk[1 : length - 2])
        devs.append((index, length, dev))
    thr = {
        key: tuple(max(scale * _pooled(chunks), floor, EPS) for chunks in pool)
        for key, pool in pools.items()
    }
    out: list[int] = []
    for index, length, dev in devs:
        flag = np.zeros(length, dtype=bool)
        for key, (resid, acc, jerk) in dev.items():
            res_thr, acc_thr, jerk_thr = thr[key]
            flag |= (resid > res_thr) & ((acc > acc_thr) | (jerk > jerk_thr))
        if flag.any():
            out.append(index)
    return out


def _gripper_dims(leaf: Path) -> dict[str, set[int]]:
    """Raw dims whose ``meta/info.json`` feature name contains ``gripper``, per source."""
    features = json.loads((leaf / "meta" / "info.json").read_text())["features"]
    out: dict[str, set[int]] = {}
    for source, col in (("state", "observation.state"), ("action", "action")):
        names = features[col].get("names")
        if isinstance(names, dict):
            names = [n for v in names.values() for n in (v if isinstance(v, list) else [v])]
        if isinstance(names, list):
            out[source] = {i for i, n in enumerate(names) if "gripper" in str(n).lower()}
    return out


def _slice(arr: np.ndarray, sl: list[int | None]) -> np.ndarray:
    out = np.zeros((len(arr), len(sl)), dtype=arr.dtype)
    for d, i in enumerate(sl):
        if i is not None:
            out[:, d] = arr[:, i]
    return out


def trained_signal(
    name: str, state: np.ndarray, action: np.ndarray, raw_skip: dict[str, set[int]]
) -> tuple[dict[str, np.ndarray], dict[str, set[int]]]:
    """State and action as the model sees them (registry slices, axis-angle → euler; no
    normalization), and the skipped dims remapped to post-slice indices."""
    from shared import axis_angle_to_euler

    from reki.data.robotics.registry import AXIS_ANGLE, DATASETS
    from reki.data.robotics.transforms import rot_blocks

    spec = DATASETS[name]
    if spec.state_slice is not None:
        state = _slice(state, spec.state_slice)
    if spec.action_slice is not None:
        action = action[:, spec.action_slice]
    blocks = rot_blocks(spec.embodiment)
    if spec.action_rot_rep == AXIS_ANGLE:
        action = axis_angle_to_euler(action, blocks)
    if spec.state_rot_rep == AXIS_ANGLE:
        state = axis_angle_to_euler(state, blocks)
    context = CONTEXT_SKIP_RAW.get(name, {})
    skip: dict[str, set[int]] = {}
    for source, sl in (("state", spec.state_slice), ("action", spec.action_slice)):
        raw = raw_skip.get(source, set()) | context.get(source, set())
        skip[source] = raw if sl is None else {d for d, i in enumerate(sl) if i in raw}
    return {"state": state, "action": action}, skip


def sudden_change(leaf: Path, name: str) -> list[int]:
    """The sudden-change episodes of one leaf of registry dataset ``name``."""
    parts = [
        read_columns(f, ["observation.state", "action", "episode_index"]) for f in data_files(leaf)
    ]
    epi = np.concatenate([p["episode_index"] for p in parts])
    signal, skip = trained_signal(
        name,
        np.concatenate([p["observation.state"] for p in parts]),
        np.concatenate([p["action"] for p in parts]),
        _gripper_dims(leaf),
    )
    starts = np.flatnonzero(np.diff(epi, prepend=epi[0] - 1))
    episodes = [
        (int(epi[s]), {k: v[s:e] for k, v in signal.items()})
        for s, e in zip(starts, [*starts[1:], len(epi)])
    ]
    return sudden_change_episodes(episodes, skip)


def ur_frozen_arm(leaf: Path, name: str) -> list[int]:
    """The episodes of one leaf whose arm joints never move."""
    std, eps = state_std(leaf)
    return [eps[i] for i in np.flatnonzero(std[:, UR_ARM_DIMS].max(1) < UR_ARM_THR)]


def scene_cuts(video: Path) -> list[float]:
    """The scene-cut times (s) of one video file."""
    import re
    import subprocess

    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(video), "-an", "-vf",
         f"scale=160:90,select='gt(scene,{CUT_SCENE})',showinfo", "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    ).stderr  # fmt: skip
    return [float(t) for t in re.findall(r"pts_time:([\d.]+)", out)]


def cut_files(
    spans: dict[Path, list[tuple[int, float, float]]], cuts: dict[Path, list[float]]
) -> list[int]:
    """The episodes of the video files (``spans``: per file ``(episode, from_s, to_s)``) where at
    least ``CUT_FILE_FRAC`` of the episodes hold an internal cut."""
    out: list[int] = []
    for video, eps in spans.items():
        with_cut = sum(
            any(t0 + CUT_EDGE_S < c < t1 - CUT_EDGE_S for c in cuts[video]) for _, t0, t1 in eps
        )
        if with_cut >= CUT_FILE_FRAC * len(eps):
            out += [ep for ep, _, _ in eps]
    return sorted(out)


def video_cuts(leaf: Path, name: str) -> list[int]:
    """The episodes of one leaf in stitched video files of its ``top`` camera."""
    from concurrent.futures import ThreadPoolExecutor

    import pyarrow as pa
    import pyarrow.parquet as pq

    from reki.data.robotics.cameras import CAMERA_KEYS, TOP
    from reki.data.robotics.registry import DATASETS

    (key,) = [k for k, v in DATASETS[name].cameras.items() if v == CAMERA_KEYS[TOP]]
    col = f"videos/{key}"
    info = json.loads((leaf / "meta" / "info.json").read_text())
    files = sorted((leaf / "meta" / "episodes").glob("**/*.parquet"))
    cols = ["episode_index", *(f"{col}/{c}" for c in ("chunk_index", "file_index"))]
    cols += [f"{col}/from_timestamp", f"{col}/to_timestamp"]
    table = pa.concat_tables([pq.read_table(f, columns=cols) for f in files])
    spans: dict[Path, list[tuple[int, float, float]]] = {}
    for ep, c, f, t0, t1 in zip(*(table.column(x).to_pylist() for x in cols)):
        rel = info["video_path"].format(video_key=key, chunk_index=c, file_index=f)
        spans.setdefault(leaf / rel, []).append((ep, t0, t1))
    with ThreadPoolExecutor(len(os.sched_getaffinity(0))) as pool:
        cuts = dict(zip(spans, pool.map(scene_cuts, spans)))
    return cut_files(spans, cuts)


def short_video(leaf: Path, name: str) -> list[int]:
    """The episodes of one leaf with fewer video frames than rows on some camera."""
    from .verify import episode_video_frames

    return sorted({ep for _, ep, n, frames, _ in episode_video_frames(leaf) if frames < n})


def failure(leaf: Path, name: str) -> list[int]:
    """The episodes of one leaf with ``is_episode_successful`` False."""
    out: set[int] = set()
    for f in data_files(leaf):
        cols = read_columns(f, ["episode_index", "is_episode_successful"])
        out |= {int(ep) for ep in np.unique(cols["episode_index"][~cols["is_episode_successful"]])}
    return sorted(out)


DETECTORS = {
    "failure": failure,
    "frozen": frozen,
    "short_video": short_video,
    "sudden_change": sudden_change,
    "ur_frozen_arm": ur_frozen_arm,
    "video_cuts": video_cuts,
}
