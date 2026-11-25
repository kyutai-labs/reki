"""Reader for Xperience-10M (``ropedia-ai/xperience-10m``, gated: manual approval + signed
agreement on HF). Never publish its caption text: the subtask lookup is rebuilt locally.

Upstream per recording ``<uuid>/<ep>/``: ``annotation.hdf5`` (~1.9 GB; native MANO hands in
``hand_mocap``, the SLAM pose in ``slam``, the rig in ``calibration``, captions in ``caption``,
per-frame device clock in ``video/device_timestamp``) and ``stereo_left.mp4`` (the rectified
left stereo camera c0: colour, 512×512 pinhole, ``calibration/cam01/K``; variable frame rate, a
25 fps grid with dropped frames, the same frame times as the fisheye videos).

Frames (HOMIE-toolkit ``data_loader.py``): ``slam`` is world→body ``T_bw``;
``calibration/cam01/T_c0_b`` is body→c0 ``B`` (per device); ``hand_mocap`` joints, translation
and global orient are in c0. The observation camera is c0, so the stored camera pose is
``(B·T_bw)⁻¹`` (cam→world) and the stored wrist is the world pose of the MANO wrist joint
(``joints_3d[:, 0]`` = translation + J0(β)) with the c0 global orient carried to world. The
loader's ``T_cam⁻¹·T_wrist`` then gives the wrist in c0 at the observation frame.

Download: the ``stereo_left.mp4`` videos as a snapshot, and from every ``annotation.hdf5``
listed at the pinned commit only the arrays the build reads (HTTP range reads) →
``upstream/xperience_annotations/<uuid>__<ep>__annotation.npz``.

Build: the recordings with a caption, sorted by that file name, go round-robin into 96 leaves
``xperience_NN`` (``XperiencePin.leaves``). Recordings without a ``slam`` pose are skipped.
Each recording is resampled to constant 20 fps: split at source gaps > ``GAP_S``, per segment a
grid of 1/20 s; the world poses by lerp (translation, finger pose) and slerp (rotations), video
by the nearest source frame, at its native 512×512, encoded with pyav (blocking, libx264). Segments < 1 s are dropped. The
frame times are the decoder PTS, so pyav/ffmpeg are pinned by the env.

Subtask lookup (``meta/subtask_lookup.jsonl``, written by ``curate``): each output frame maps through its source frame's
device time to the caption "Current Action" that contains it; runs of one label are intervals.
The intervals use the demuxed PTS (packet order), unless a leaf's episode count under them
differs from the build's: then that leaf uses the decoder PTS of the build.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
from collections.abc import Callable, Generator, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from ..common import v30
from ..common.download import RangeFile, listing, upstream_dir
from ..common.lookup import runs
from ..pins import HfSnapshot, Pin
from ..recipe import Recipe
from . import egocentric as ego

logger = logging.getLogger(__name__)

REPO = "ropedia-ai/xperience-10m"
REVISION = "ce943cf271a758b60240084892d05cf6dc12dd90"
VIDEO = "stereo_left.mp4"
SOURCE = HfSnapshot(REPO, REVISION, allow=(f"*/*/{VIDEO}",))
SUBSET = (
    "3efd6a52-1457-4437-a3bf-9d5894614fe6/ep4/annotation.hdf5",
    "580694cd-1b3b-4192-ac5b-3f549d5fb727/ep19/annotation.hdf5",
    "cdc1ae12-a460-48ac-a892-7d314095c4b1/ep23/annotation.hdf5",
    "cf0c4437-edb2-4455-9746-2a759801116f/ep6/annotation.hdf5",
    "0ecb6a4b-7479-4c64-bc8d-e4e2d3d17311/ep1/annotation.hdf5",
)
"""``--subset`` recordings, in 2 leaves. Leaf 0: ``3efd6a52/ep4`` and ``cdc1ae12/ep23``; the
latter gives an episode under the demuxed PTS but none under the decoder's (fewer decodable
frames), so the leaf's lookup falls back to the decoder PTS. Leaf 1: ``580694cd/ep19`` (lookup
from the demuxed PTS, whose episode is one frame longer) and ``cf0c4437/ep6`` (no ``slam``).
``0ecb6a4b/ep1`` has no caption."""


@dataclass(frozen=True)
class XperiencePin(Pin):
    """``annotations``: the ``annotation.hdf5`` files read; None: all at ``REVISION``."""

    annotations: tuple[str, ...] | None = None
    leaves: int = ego.LEAVES

    def annotation_files(self, root: Path) -> list[str]:
        if self.annotations is not None:
            return sorted(self.annotations)
        files = listing(HfSnapshot(REPO, REVISION), upstream_dir(root, SOURCE))
        return sorted(f for f in files if f.endswith("/annotation.hdf5"))


PIN = XperiencePin(
    (SOURCE,),
    episodes=13_313,
    frames=70_508_508,
    subset=XperiencePin(
        (HfSnapshot(REPO, REVISION, tuple(f"{r.rsplit('/', 1)[0]}/{VIDEO}" for r in SUBSET)),),
        episodes=2,
        frames=11_932,
        annotations=SUBSET,
        leaves=2,
    ),
)
FPS = 20
IMG_W, IMG_H = 512, 512
GAP_S = 0.4
MIN_FRAMES = FPS
SIDES = ("left", "right")
NO_CAPTION, NO_SLAM, OK = "no_caption", "no_slam", "ok"


def sidecar_name(rel: str) -> str:
    """``<uuid>/<ep>/annotation.hdf5`` → ``<uuid>__<ep>__annotation.npz`` (the sort key)."""
    return rel.replace("/", "__")[: -len(".hdf5")] + ".npz"


# ------------------------------------------------------------------ annotation sweep


def finger_pose(rotmats: np.ndarray) -> np.ndarray:
    """``[T, 15, 3, 3]`` MANO finger rotations → ``[T, 45]`` axis-angle (untracked joints 0)."""
    valid = ego.valid_so3(rotmats)
    aa = np.zeros((len(rotmats), 15, 3), np.float64)
    if valid.any():
        aa[valid] = Rotation.from_matrix(rotmats[valid]).as_rotvec()
    return aa.reshape(len(rotmats), 45)


INT64 = range(-(2**63), 2**63)


def _frame(value: Any) -> int:
    """A caption frame value (device ns) as an int; a non-integer raises."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"caption frame {value!r} is not an integer")
    return int(value)


def current_actions(
    caption: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int, str]]]:
    """``(start_ns, end_ns, label, dropped)`` of every labelled caption "Current Action".

    A caption with a frame outside int64 gives no actions; ``dropped`` lists them all. Frames
    outside the recording are kept."""
    acts = [
        (_frame(a["start_frame"]), _frame(a["end_frame"]), str(a.get("label", "")).strip())
        for seg in caption.get("segments") or []
        for a in seg.get("Current Action") or []
    ]
    acts = [a for a in acts if a[2]]
    dropped: list[tuple[int, int, str]] = []
    if any(a[0] not in INT64 or a[1] not in INT64 for a in acts):
        acts, dropped = [], acts
    return (
        np.array([a[0] for a in acts], np.int64),
        np.array([a[1] for a in acts], np.int64),
        np.array([a[2] for a in acts], dtype=str),
        dropped,
    )


def recording(name: str) -> str:
    """A sidecar name's ``<uuid>/<ep>``."""
    return "/".join(name.split("__")[:2])


def sweep_file(rel: str, out: Path) -> list[tuple[int, int, str]]:
    """Range-read one ``annotation.hdf5`` into its sidecar npz (``status``: ok/no_caption/no_slam);
    returns the actions ``current_actions`` dropped."""
    import h5py  # type: ignore[import-not-found]
    from huggingface_hub import get_token

    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{rel}"
    rec: dict[str, Any] = {}
    dropped: list[tuple[int, int, str]] = []
    with h5py.File(RangeFile(url, {"Authorization": f"Bearer {get_token()}"}), "r") as h:
        if "caption" not in h:
            rec["status"] = NO_CAPTION
        else:
            caption = h["caption"][()]
            caption = json.loads(caption.decode() if isinstance(caption, bytes) else caption)
            rec["main_task"] = str(caption.get("config", {}).get("Main Task", "")).strip()
            rec["act_start"], rec["act_end"], rec["act_label"], dropped = current_actions(caption)
            rec["device_ts"] = (
                np.asarray(h["video/device_timestamp"][:]).astype("U").astype(np.int64)
            )
            for side in SIDES:
                pose = h[f"hand_mocap/{side}_mano_hand_pose"][:].reshape(-1, 15, 3, 3)
                orient = h[f"hand_mocap/{side}_mano_hand_global_orient"][:].reshape(-1, 3, 3)
                valid = ego.valid_so3(orient)
                rec[f"{side}_finger_aa"] = finger_pose(pose).astype(np.float32)
                rec[f"{side}_wrist_aa"] = np.where(
                    valid[:, None],
                    ego.rotmat_to_rotvec(np.where(valid[:, None, None], orient, np.eye(3))),
                    0,
                ).astype(np.float32)
                rec[f"{side}_wrist_c0"] = h[f"hand_mocap/{side}_joints_3d"][:, 0].astype(np.float32)
                rec[f"{side}_valid"] = valid
            if "slam" not in h:
                rec["status"] = NO_SLAM
            else:
                rec["status"] = OK
                rec["slam_quat_wxyz"] = np.asarray(h["slam/quat_wxyz"][:], np.float64)
                rec["slam_trans_xyz"] = np.asarray(h["slam/trans_xyz"][:], np.float64)
                rec["T_c0_b"] = np.asarray(h["calibration/cam01/T_c0_b"][()], np.float64)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, **rec)
    tmp.rename(out)
    return dropped


def sweep(files: list[str], out: Path, workers: int = 16) -> None:
    """Sidecars of ``files`` into ``out`` by ``workers`` processes (h5py serializes threads);
    existing ones are kept (resumable). Each dropped caption action is logged."""
    out.mkdir(parents=True, exist_ok=True)
    todo = [rel for rel in files if not (out / sidecar_name(rel)).exists()]
    logger.info("xperience: %d of %d annotation files to read", len(todo), len(files))
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=ctx) as ex:
        outs = [out / sidecar_name(rel) for rel in todo]
        for i, (rel, dropped) in enumerate(zip(todo, ex.map(sweep_file, todo, outs))):
            for start, end, label in dropped:
                logger.warning(
                    "xperience: %s: caption action [%d, %d] %r dropped (a caption frame is outside int64)",
                    recording(sidecar_name(rel)),
                    start,
                    end,
                    label,
                )
            if (i + 1) % 200 == 0:
                logger.info("xperience: %d / %d annotation files", i + 1, len(todo))


# ------------------------------------------------------------------ frames


def world_poses(ann: dict) -> dict[str, np.ndarray]:
    """Per source frame, the c0 camera pose (cam→world ``camera_t``/``camera_aa``) and each
    hand's world wrist (``{side}_wrist_t``/``{side}_wrist_aa``; untracked frames are garbage
    and masked by ``{side}_valid`` downstream)."""
    quat = np.asarray(ann["slam_quat_wxyz"], np.float64)
    if (np.abs(np.linalg.norm(quat, axis=1) - 1.0) > 1e-3).any():
        raise ValueError("slam/quat_wxyz has a non-unit quaternion")
    B = np.asarray(ann["T_c0_b"], np.float64)
    r_cw = Rotation.from_matrix(B[:3, :3]) * Rotation.from_quat(quat[:, [1, 2, 3, 0]])
    t_cw = np.asarray(ann["slam_trans_xyz"], np.float64) @ B[:3, :3].T + B[:3, 3]
    r_wc = r_cw.inv()
    t_wc = -r_wc.apply(t_cw)
    out = {"camera_t": t_wc, "camera_aa": r_wc.as_rotvec()}
    for side in SIDES:
        out[f"{side}_wrist_t"] = r_wc.apply(np.asarray(ann[f"{side}_wrist_c0"], np.float64)) + t_wc
        out[f"{side}_wrist_aa"] = (
            r_wc * Rotation.from_rotvec(np.asarray(ann[f"{side}_wrist_aa"], np.float64))
        ).as_rotvec()
    return out


# ------------------------------------------------------------------ resampling


def decoder_pts(mp4: Path, n: int) -> np.ndarray:
    """Zero-origin times (s) of the first ``n`` decoded frames (some videos decode fewer)."""
    import av

    pts: list[int | None] = []
    time_base: Fraction | None = None
    with av.open(str(mp4)) as c:
        time_base = c.streams.video[0].time_base
        for frame in c.decode(video=0):
            pts.append(frame.pts)
            if len(pts) >= n:
                break
    assert time_base is not None
    out = np.asarray(pts, np.float64) * float(time_base)
    return out - out[0]


def demux_pts(mp4: Path, n: int) -> np.ndarray:
    """Zero-origin times (s) of the first ``n`` packets in PTS order (no decode)."""
    import av

    pts: list[int] = []
    time_base: Fraction | None = None
    with av.open(str(mp4)) as c:
        stream = c.streams.video[0]
        time_base = stream.time_base
        pts += [p.pts for p in c.demux(stream) if p.pts is not None]
    assert time_base is not None
    out = np.sort(np.asarray(pts, np.float64))[:n] * float(time_base)
    return out - out[0]


def segments(pts: np.ndarray) -> Iterator[tuple[int, int, np.ndarray]]:
    """``(a, b, grid)`` per kept segment: source frames ``[a, b)`` between gaps > ``GAP_S`` and
    the output times; clips shorter than ``MIN_FRAMES`` source frames give none."""
    if len(pts) < MIN_FRAMES:
        return
    cuts = [0, *(np.flatnonzero(np.diff(pts) > GAP_S) + 1).tolist(), len(pts)]
    for a, b in zip(cuts[:-1], cuts[1:]):
        t0, t1 = pts[a], pts[b - 1]
        grid = np.minimum(t0 + np.arange(int((t1 - t0) * FPS) + 1) / FPS, t1)
        if len(grid) >= MIN_FRAMES:
            yield a, b, grid


def _bracket(p: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.searchsorted(p, grid).clip(1, len(p) - 1)


def nearest_source(pts: np.ndarray, a: int, b: int, grid: np.ndarray) -> np.ndarray:
    """The source frame nearest to each output time."""
    p = pts[a:b]
    hi = _bracket(p, grid)
    lo = hi - 1
    return np.where(np.abs(p[lo] - grid) <= np.abs(p[hi] - grid), lo, hi) + a


def resample_hand(
    ann: dict, side: str, pca: ego.Eigengrasps, pts: np.ndarray, a: int, b: int, grid: np.ndarray
) -> np.ndarray:
    """One hand's ``[N, 16]`` action block on the output grid."""
    p = pts[a:b]
    transl = np.stack(
        [np.interp(grid, p, c) for c in ann[f"{side}_wrist_t"][a:b].astype(np.float64).T], 1
    )
    fingers = np.stack(
        [np.interp(grid, p, c) for c in ann[f"{side}_finger_aa"][a:b].astype(np.float64).T], 1
    )
    rot = Slerp(p, Rotation.from_rotvec(ann[f"{side}_wrist_aa"][a:b].astype(np.float64)))(grid)
    rotvec = ego.rotmat_to_rotvec(rot.as_matrix())
    tracked = ann[f"{side}_valid"][a:b].astype(bool)
    hi = _bracket(p, grid)
    return ego.hand_columns(transl, rotvec, pca.transform(fingers), tracked[hi - 1] & tracked[hi])


def resample_camera(ann: dict, pts: np.ndarray, a: int, b: int, grid: np.ndarray) -> np.ndarray:
    p = pts[a:b]
    transl = np.stack([np.interp(grid, p, c) for c in ann["camera_t"][a:b].astype(np.float64).T], 1)
    rot = Slerp(p, Rotation.from_rotvec(ann["camera_aa"][a:b].astype(np.float64)))(grid).as_rotvec()
    return ego.camera_state(transl, rot)


def subtask_intervals(ann: dict, pts: np.ndarray, a: int, b: int, grid: np.ndarray) -> list[list]:
    """``[start, end, label]`` runs over the output frames (end-exclusive; frames in no action
    are not covered)."""
    ns = ann["device_ts"][nearest_source(pts, a, b, grid)]
    labels: list[str | None] = [None] * len(grid)
    for s, e, label in zip(ann["act_start"], ann["act_end"], ann["act_label"]):
        for k in np.flatnonzero((ns >= s) & (ns < e)):
            labels[k] = str(label)
    return [list(run) for run in runs(labels)]


def clip_lookup(ann: dict, pts: np.ndarray, n_poses: int) -> list[dict]:
    """The lookup entries (without ``episode_index``) of one recording's episodes."""
    if len(pts) < MIN_FRAMES:
        return []
    device_ts = ann["device_ts"]
    if len(device_ts) < len(pts) or abs(len(device_ts) - n_poses) > 2:
        raise ValueError(f"{len(device_ts)} device times for {len(pts)} frames / {n_poses} poses")
    task = str(ann["main_task"])
    out = []
    for a, b, grid in segments(pts):
        entry: dict = {
            "intervals": subtask_intervals(ann, pts, a, b, grid) if len(ann["act_start"]) else []
        }
        if task:
            entry["task"] = task
        out.append(entry)
    return out


def encode(frames: list[np.ndarray], path: Path) -> None:
    """Blocking libx264 encode of RGB frames at constant ``FPS`` (pts = frame number)."""
    import av

    with av.open(str(path), mode="w") as c:
        stream = c.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = IMG_W, IMG_H, "yuv420p"
        stream.time_base = Fraction(1, FPS)
        stream.thread_count = 1
        stream.options = {"crf": "23", "preset": "veryfast"}
        for k, arr in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24").reformat(format="yuv420p")
            frame.pts = k
            for packet in stream.encode(frame):
                c.mux(packet)
        for packet in stream.encode():
            c.mux(packet)


def _frames(mp4: Path) -> Generator[np.ndarray, None, None]:
    import av

    with av.open(str(mp4)) as c:
        for frame in c.decode(video=0):
            yield frame.reformat(width=IMG_W, height=IMG_H, format="rgb24").to_ndarray()


# ------------------------------------------------------------------ reader


class Xperience(ego.EgoReader):
    prefix = "xperience"

    def download(self, name: str, root: Path, pin: Pin) -> None:
        assert isinstance(pin, XperiencePin)
        super().download(name, root, pin)
        sweep(pin.annotation_files(root), annotations_dir(root))

    def _statuses(self, root: Path, pin: XperiencePin) -> dict[str, str]:
        """Per recording (sidecar name): its sweep ``status``."""
        statuses = {}
        for rel in pin.annotation_files(root):
            with np.load(annotations_dir(root) / sidecar_name(rel)) as d:
                statuses[sidecar_name(rel)] = str(d["status"])
        return statuses

    def _clips(self, root: Path, pin: XperiencePin) -> list[list[str]]:
        """Per leaf, its recordings (sidecar names) with a caption and a ``slam`` camera."""
        statuses = self._statuses(root, pin)
        names = sorted(n for n, s in statuses.items() if s != NO_CAPTION)
        return [
            [n for n in names[leaf :: pin.leaves] if statuses[n] == OK]
            for leaf in range(min(pin.leaves, len(names)))
        ]

    def _episodes(self, clips: list[str], root: Path, work: Path) -> Iterator[v30.Episode]:
        pca = ego.Eigengrasps.load()
        for clip in clips:
            ann = load_annotation(root, clip)
            ann.update(world_poses(ann))
            mp4 = video_path(root, clip)
            pts = decoder_pts(mp4, len(ann["left_finger_aa"]))
            task = str(ann["main_task"]) or ego.FALLBACK_TASK
            frames = _frames(mp4)
            index, frame = -1, None
            if next(segments(pts), None) is None:
                logger.info("xperience: %s skipped (no decodable episode)", recording(clip))
            for a, b, grid in segments(pts):
                action = np.concatenate(
                    [resample_hand(ann, side, pca, pts, a, b, grid) for side in SIDES], axis=1
                )
                state = resample_camera(ann, pts, a, b, grid)
                picked = []
                for src in nearest_source(pts, a, b, grid):
                    while index < src:
                        index, frame = index + 1, next(frames)
                    picked.append(frame)
                path = work / f"{clip[: -len('.npz')]}__{a}.mp4"
                encode(picked, path)
                yield v30.Episode(a, state, action, task, video_src={ego.VIDEO_KEY: path})
            frames.close()

    def build(self, name: str, root: Path, pin: Pin, shard: tuple[int, int]) -> None:
        assert isinstance(pin, XperiencePin)
        i, n = shard
        for clip, status in self._statuses(root, pin).items():
            if status != OK:
                logger.info("xperience: %s skipped (%s)", recording(clip), status.replace("_", " "))
        for leaf, clips in list(enumerate(self._clips(root, pin)))[i::n]:
            self.write_leaf(name, root, leaf, partial(self._episodes, clips, root), FPS)

    def curate(self, name: str, root: Path, pin: Pin) -> None:
        """Each leaf's ``meta/subtask_lookup.jsonl``; the leaf's episode count is the gate."""
        assert isinstance(pin, XperiencePin)
        for leaf, clips in enumerate(self._clips(root, pin)):
            out = self.leaf_dir(name, root, leaf)
            if not out.exists():
                continue
            built = json.loads((out / "meta" / "info.json").read_text())["total_episodes"]
            source = "demuxed"
            entries = leaf_lookup(root, clips, demux_pts)
            if len(entries) != built:
                source = "decoder"
                entries = leaf_lookup(root, clips, decoder_pts)
            if len(entries) != built:
                raise ValueError(f"{out.name}: {len(entries)} lookup entries, {built} episodes")
            lines = [json.dumps({"episode_index": e, **entry}) for e, entry in enumerate(entries)]
            (out / "meta" / "subtask_lookup.jsonl").write_text("\n".join(lines) + "\n")
            covered = sum(bool(entry["intervals"]) for entry in entries)
            logger.info(
                "%s: lookup from %s PTS, %d/%d episodes covered", out.name, source, covered, built
            )


def annotations_dir(root: Path) -> Path:
    return root / "upstream" / "xperience_annotations"


def load_annotation(root: Path, clip: str) -> dict:
    with np.load(annotations_dir(root) / clip) as d:
        return {k: d[k] for k in d.files}


def video_path(root: Path, clip: str) -> Path:
    uuid, ep, _ = clip.split("__")
    return upstream_dir(root, SOURCE) / uuid / ep / VIDEO


def leaf_lookup(
    root: Path, clips: list[str], pts_fn: Callable[[Path, int], np.ndarray]
) -> list[dict]:
    """The lookup entries of a leaf's episodes with the frame times of ``pts_fn``."""
    out: list[dict] = []
    for clip in clips:
        ann = load_annotation(root, clip)
        n_poses = len(ann["left_finger_aa"])
        out += clip_lookup(ann, pts_fn(video_path(root, clip), n_poses), n_poses)
    return out


PINS = {"xperience": PIN}
RECIPES = {
    "xperience": Recipe(
        Xperience(),
        curate=(),
        stats=(),
        horizon=(ego.ego_stats, partial(ego.ego_wrist, windows=6)),
    )
}
