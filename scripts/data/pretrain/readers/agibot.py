"""Reader for AgiBot World 2026 (``ImitationLearning``, 89 shards at the pin): one leaf.

Each shard is a tar.gz of one LeRobot v2.1 dataset whose episodes are long recordings. Per
shard: extract, split each recording at its ``Task Frame`` annotations (``info.json``
``key_frame``, as upstream's ``split_episodes_tool``), one episode per Task Frame, in shard
order then frame order. Per episode:

- state: the shard's ``observation.state`` fields scattered into the 169-dim layout
  ``STATE_FIELDS`` (162-dim shards lack ``state/robot/{position,orientation}``: zeros);
- task: the Task Frame comment (text rules, ``common.tasks``); its instruction segments,
  clipped to the episode, go to ``meta/instruction_segments.jsonl`` (lookup at curate);
- videos: each camera re-sliced to the episode's frames with ffmpeg (``ENCODE``). Resolution
  fix: a camera at 2× its ``CAMERA_SIZES`` size (task_3477 hand and fisheye cameras) is scaled
  down in the same pass; any other size raises.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..common import lookup, v30
from ..common.download import untar
from ..common.stats import op
from ..common.tasks import TextRules
from ..common.video import ffprobe
from ..pins import HfSnapshot, Pin
from ..recipe import Recipe

NAME = "agibot_world_2026"
REPO = "agibot-world/AgiBotWorld2026"
REVISION = "21eb51a5f9285c5433bc2e8d473e712c11ccd9ee"
LEAF = "agibot_world_2026_lerobot_v30"
FPS = 30
ROBOT_TYPE = "g2a"

SHARDS = (
    "ImitationLearning/CommercialSpaces/task_3400/313498_314085.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3400/314096_314575.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3400/314576_315886.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3400/315903_317024.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3400/317029_317933.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3400/346206_347749.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3401/352507_353983.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3401/353984_357369.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3401/357385_358520.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3401/358525_362596.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3401/363333_364278.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3401/364289_366221.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3401/366235_366995.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3401/366997_368165.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3401/368167_368944.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3401/368945_369053.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3401/369057_369226.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3401/369229_370144.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3401/370198_371077.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3401/371092_371850.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3401/371906_372880.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3401/372890_399007.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3401/399093_399454.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3402/373661_375235.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3402/375260_377477.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3402/377489_380881.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3402/380892_384933.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3402/384954_385428.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3404/305627_321730.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3404/321801_329470.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3404/374845_377721.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3404/377745_378588.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3404/389330_396655.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3405/389111_389369.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3477/248923_250431.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3477/250434_252697.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3477/252787_255587.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3641/304773_306558.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3641/306569_307774.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3641/307787_308785.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3641/308789_310351.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3641/312244_314720.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3641/314721_317680.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3641/317686_319845.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3641/319854_320987.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3641/321004_322875.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3641/322888_323129.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3641/323131_324270.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3641/324278_324936.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3641/324949_325550.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3641/325556_326880.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3641/326887_327595.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3641/327603_328738.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3641/328740_329556.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3641/329558_329877.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3641/329879_330176.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3641/330177_330503.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3641/330504_333131.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3641/333145_338811.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3641/338871_343681.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3641/343691_352441.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3705/312820_313211.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3777/325306_327364.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3777/327380_328715.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3777/328722_329585.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3777/329589_330129.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3777/330132_331330.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3777/331358_333284.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3777/333286_336077.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3777/336089_338299.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3777/338316_342675.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3777/342697_352057.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3777/352058_353519.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3777/353521_357589.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3777/357780_362729.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3777/362736_365107.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3777/365112_368927.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3777/368929_370517.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3777/370529_373068.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3777/373071_374770.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3777/374778_377296.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3777/377297_378412.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3777/378413_380089.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3777/380090_381352.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3777/381367_384922.tar.gz",
    "ImitationLearning/CommercialSpaces/task_3777/384931_386178.tar.gz",
    "ImitationLearning/CommercialSpaces/task_4053/368961_369296.tar.gz",
    "ImitationLearning/CommercialSpaces/task_4053/369297_371773.tar.gz",
    "ImitationLearning/CommercialSpaces/task_4053/371782_373846.tar.gz",
)
"""Every ImitationLearning shard at ``REVISION``, in build order."""
SUBSET_SHARD = "ImitationLearning/CommercialSpaces/task_3477/252787_255587.tar.gz"
"""72 episodes, all with 2×-resolution hand/fisheye cameras."""

STATE_FIELDS = (
    ("state/left_effector/position", 1),
    ("state/right_effector/position", 1),
    ("state/end/wrench", 12),
    ("state/end/velocity", 12),
    ("state/end/arm_orientation", 8),
    ("state/end/arm_position", 6),
    ("state/joint/position", 14),
    ("state/joint/effort", 14),
    ("state/joint/velocity", 14),
    ("state/head/position", 3),
    ("state/waist/position", 5),
    ("state/robot/position", 3),
    ("state/robot/orientation", 4),
    ("extrinsic_end_T_hand_left_rgbd_aligned/rotation_matrix", 9),
    ("extrinsic_end_T_hand_right_rgbd_aligned/rotation_matrix", 9),
    ("extrinsic_end_T_head_left_fisheye_aligned/rotation_matrix", 9),
    ("extrinsic_end_T_head_right_fisheye_aligned/rotation_matrix", 9),
    ("extrinsic_end_T_head_front_rgbd_aligned/rotation_matrix", 9),
    ("extrinsic_end_T_head_back_fisheye_aligned/rotation_matrix", 9),
    ("extrinsic_end_T_hand_left_rgbd_aligned/translation_vector", 3),
    ("extrinsic_end_T_hand_right_rgbd_aligned/translation_vector", 3),
    ("extrinsic_end_T_head_left_fisheye_aligned/translation_vector", 3),
    ("extrinsic_end_T_head_right_fisheye_aligned/translation_vector", 3),
    ("extrinsic_end_T_head_front_rgbd_aligned/translation_vector", 3),
    ("extrinsic_end_T_head_back_fisheye_aligned/translation_vector", 3),
)
"""The 169-dim state layout: every field of the 169-dim shards, in their order."""

CAMERA_SIZES = {
    "observation.images.top_head": (640, 400),
    "observation.images.hand_left": (640, 528),
    "observation.images.hand_right": (640, 528),
    "observation.images.head_left_fisheye": (960, 768),
    "observation.images.head_right_fisheye": (960, 768),
    "observation.images.head_back_fisheye": (960, 768),
    "observation.images.head_depth": (640, 400),
}
"""Camera → (width, height) of the leaf's videos."""

ENCODE = ["-c:v", "libsvtav1", "-preset", "8", "-crf", "35", "-pix_fmt", "yuv420p"]
ENCODE_DEPTH = ["-c:v", "png", "-pix_fmt", "gray16be"]
"""Per-episode re-slice settings (the trained data's: libsvtav1 defaults; depth maps as png)."""
WORKERS = max(1, len(os.sched_getaffinity(0)) // 8)
"""Concurrent ffmpeg encodes (each svt-av1 encode is multithreaded)."""


def source(shards: tuple[str, ...]) -> HfSnapshot:
    return HfSnapshot(REPO, REVISION, shards)


PINS = {
    NAME: Pin(
        (source(SHARDS),),
        episodes=21650,
        frames=17390604,
        subset=Pin((source((SUBSET_SHARD,)),), episodes=72, frames=143641),
    ),
}


# ------------------------------------------------------------------ episodes of one shard


def task_frames(info: dict) -> list[dict]:
    """The Task Frame segments ``{episode, start, end, comment}`` of a shard, in recording then
    frame order."""
    out = []
    for ep in sorted(info.get("key_frame", {}), key=int):
        frames = [
            f
            for f in info["key_frame"][ep].get("dual", [])
            if f.get("frame_type_name") == "Task Frame"
        ]
        for f in sorted(frames, key=lambda f: f["start"]):
            comment = f.get("frame_detail", {}).get("comment", "")
            out.append(
                {"episode": int(ep), "start": f["start"], "end": f["end"], "comment": comment}
            )
    return out


def scatter_state(state: np.ndarray, info: dict) -> np.ndarray:
    """A shard's ``[N, D]`` state in the ``STATE_FIELDS`` layout (absent fields zero)."""
    fields = info["features"]["observation.state"]["field_descriptions"]
    out = np.zeros((len(state), sum(d for _, d in STATE_FIELDS)), dtype=state.dtype)
    offset = 0
    for name, dim in STATE_FIELDS:
        indices = fields.get(name, {}).get("indices") or []
        if indices:
            if len(indices) != dim:
                raise ValueError(f"state field {name}: {len(indices)} dims, layout {dim}")
            out[:, offset : offset + dim] = state[:, indices]
        offset += dim
    unknown = set(fields) - {n for n, _ in STATE_FIELDS}
    if unknown:
        raise KeyError(f"state fields outside the layout: {sorted(unknown)}")
    return out


def clip_segments(info: dict, frame: dict) -> list[dict]:
    """The recording's instruction segments that overlap the Task Frame, in episode frames."""
    out = []
    for s in info.get("instruction_segments", {}).get(str(frame["episode"]), []):
        start = max(s.get("start_frame_index", s.get("start", 0)), frame["start"])
        end = min(s.get("end_frame_index", s.get("end", 0)), frame["end"])
        if start < end:
            out.append(
                {
                    "skill": s.get("skill", ""),
                    "instruction": s.get("instruction", ""),
                    "start": start - frame["start"],
                    "end": end - frame["start"],
                }
            )
    return out


def video_size(path: Path) -> tuple[int, int]:
    w, h = ffprobe(path, "-show_entries", "stream=width,height", "-of", "csv=p=0").split(",")
    return int(w), int(h)


def rescale(size: tuple[int, int], target: tuple[int, int]) -> bool:
    """Whether a video of ``size`` is scaled to ``target``: False at ``target``, True at exactly
    2× (the resolution defect); any other size raises."""
    if size == target:
        return False
    if size == (2 * target[0], 2 * target[1]):
        return True
    raise ValueError(f"video size {size}: neither {target} nor 2×")


def slice_video(src: Path, dst: Path, start: int, end: int, key: str) -> None:
    """Frames ``[start, end)`` of ``src`` re-encoded to ``dst`` at ``CAMERA_SIZES[key]``."""
    target = CAMERA_SIZES[key]
    scale = ["-vf", f"scale={target[0]}:{target[1]}"] if rescale(video_size(src), target) else []
    codec = ENCODE_DEPTH if key.endswith("_depth") else ENCODE
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start / FPS:.6f}", "-i", str(src)]
    cmd += ["-t", f"{(end - start) / FPS:.6f}", *scale, *codec, "-an", str(dst)]
    subprocess.run(cmd, check=True, capture_output=True)


def shard_episodes(
    src: Path, work: Path, pool: ThreadPoolExecutor, rules: TextRules
) -> Iterator[tuple[v30.Episode, list[dict]]]:
    """The episodes of one extracted shard and their instruction segments; videos are sliced
    into ``work`` by ``pool``."""
    info = json.loads((src / "meta" / "info.json").read_text())
    if info["fps"] != FPS:
        raise ValueError(f"{src}: fps {info['fps']}")
    frames = task_frames(info)
    if not frames:
        raise ValueError(f"{src}: no Task Frame annotations")
    keys = v30.video_keys(info)
    if set(keys) != set(CAMERA_SIZES):
        raise KeyError(f"{src}: cameras {sorted(keys)}")
    jobs: list[dict[str, tuple[Path, Future]]] = []
    for i, frame in enumerate(frames):
        cams = {}
        for key, video in v30.v2x_files(src, info, frame["episode"])[1].items():
            dst = work / key / f"{i:06d}.mp4"
            cams[key] = (
                dst,
                pool.submit(slice_video, video, dst, frame["start"], frame["end"], key),
            )
        jobs.append(cams)
    tables: dict[int, dict[str, np.ndarray]] = {}
    for frame, cams in zip(frames, jobs):
        ep = frame["episode"]
        if ep not in tables:
            path, _ = v30.v2x_files(src, info, ep)
            tables[ep] = v30.read_columns(path, [*v30.STATE_ACTION, "frame_index"])
        cols = tables[ep]
        mask = (cols["frame_index"] >= frame["start"]) & (cols["frame_index"] < frame["end"])
        if not mask.any():
            raise ValueError(f"{src}: empty Task Frame {frame}")
        for _, future in cams.values():
            future.result()
        yield (
            v30.Episode(
                frame["episode"],
                scatter_state(cols["observation.state"][mask], info),
                cols["action"][mask],
                rules.task(frame["comment"]),
                video_src={key: dst for key, (dst, _) in cams.items()},
            ),
            clip_segments(info, frame),
        )


class AgiBot(lookup.TarballLeaf):
    leaf, fps, robot_type = LEAF, FPS, ROBOT_TYPE

    def tarball_episodes(
        self, name: str, archive: Path, work: Path, rules: TextRules
    ) -> Iterator[tuple[v30.Episode, list[dict]]]:
        """Extracted into ``work/extract`` (removed after the shard), videos sliced into
        ``work/videos``."""
        untar(archive, work / "extract", fresh=True)
        with ThreadPoolExecutor(WORKERS) as pool:
            yield from shard_episodes(work / "extract" / "data", work / "videos", pool, rules)
        shutil.rmtree(work / "extract")


RECIPES = {
    NAME: Recipe(AgiBot(), curate=("frozen", "short_video"), stats=(op("kept_qk", exact=True),))
}
