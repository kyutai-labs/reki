"""RoboMIND (``x-humanoid-robomind/RoboMIND`` at 6b43cf9a): one leaf per embodiment,
``robomind_lerobot/robomind_<embodiment>_lerobot_v30``.

Upstream: per embodiment and benchmark, one tar.gz per task folder, split into parts
(``<task>.tar.gz.part-aa`` …), each trajectory a ``trajectory.hdf5``. ``pins_robomind.tsv``
pins the 433 trained archives (embodiment, benchmark, folder, parts, episodes, trained task
text where the text rule gives another). Left out, as trained: 7 franka_3rgb ``2024_09_20_*``
archives (no ``camera_top``), gello ``place_yellow_banana_on_plastic_plate`` (truncated
upstream), gello ``clean_table_3_241210`` and ``place_plate_table_241211_12``, prod1
``pick_up_paper_ball_place_in_dustbin`` and ``pick_up_paper_cup_place_in_dustbin``. Archives
are read in (benchmark, task) order, trajectories in path order; up to ``EXTRACT_AHEAD``
archives are extracted at once, ahead of the one being encoded, into
``<root>/.work/robomind/<embodiment>/`` (about their size) and each is deleted after.

Per trajectory: state = ``puppet/<keys>``; action = ``<source>/<keys>`` (``EMBODIMENTS``), with
``shift`` 1 the next frame's puppet value (``action[t] = puppet[t+1]``, the last frame goes).
Left out, as in the trained data: a trajectory shorter than 50 frames (before the shift), one
whose JPEG frames change shape or do not decode, and (UR) an idle one: the summed per-dim
variance of its float32 state below ``IDLE_VAR``. A camera with fewer frames than the state
shortens the episode. Frames are encoded with the trained settings (``X264``). Any other read
error raises.

Task text: the folder's ``RoboMIND_v1_2_instr.csv`` instruction (else the folder name), through
the RoboMIND text rule (``TASKS``: the trained text where it differs). The one-word AgileX
folders (``BROKEN_TASKS``) take the ``task_summary`` of
``language_description_annotation_json/h5_agilex_3rgb.json`` whose ``id`` is the trajectory's
own archive dir; a trajectory without one is left out. Upstream stores 15 39_puttomato
trajectories twice, under two ids; at most one id of a pair is annotated, so the other copy is
left out. Copies in both benchmarks are dedup duplicates (same text: the first is kept).

Curation: dedup, ``frozen`` (UR: also ``ur_frozen_arm``). No subtask lookup: the datasets train
with their task text (registry ``subtask_lookup``).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import tarfile
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..common import v30
from ..common.download import ConcatFile, upstream_dir
from ..common.stats import op
from ..common.tasks import ROBOMIND, robomind_annotation, text_rules
from ..common.video import FrameEncoder
from ..pins import HfSnapshot, Pin
from ..recipe import Reader, Recipe, root_dir

logger = logging.getLogger(__name__)

EXTRACT_AHEAD = 4
"""Archives extracted at once, ahead of the one being encoded (tar.gz reads one core each)."""
REPO = "x-humanoid-robomind/RoboMIND"
REVISION = "6b43cf9a9f40d97af7e9ab8be2b86525b66b5b30"
FPS = 30
MIN_FRAMES = 50
IDLE_VAR = 1e-6
"""UR episodes whose state variance (float32, summed over dims) is below this are left out."""
X264 = (
    "-c:v",
    "libx264",
    "-preset",
    "veryfast",
    "-crf",
    "23",
    "-pix_fmt",
    "yuv420p",
    "-threads",
    "1",
)
INSTRUCTIONS = "static/RoboMIND_v1_2_instr.csv"
ANNOTATIONS = "static/language_description_annotation_json/h5_agilex_3rgb.json"
BROKEN_TASKS = (
    "35_putcarrot",
    "36_putpepper",
    "37_putegg",
    "39_puttomato",
    "40_putavocado",
    "41_putplum",
    "5_eggoven_2",
)
"""AgileX folders whose instruction is one word: the episodes take their own annotation."""
PINS_TSV = Path(__file__).parents[1] / "pins_robomind.tsv"


@dataclass(frozen=True)
class Archive:
    embodiment: str
    benchmark: str
    folder: str
    parts: int
    episodes: int
    task: str
    """The trained task text where the text rule gives another; empty: the rule."""

    def files(self) -> tuple[str, ...]:
        return part_files(self.benchmark, self.embodiment, self.folder, self.parts)


def part_files(benchmark: str, emb: str, folder: str, parts: int) -> tuple[str, ...]:
    """The first ``parts`` part files (``…tar.gz.part-aa`` on) of an archive."""
    names = [a + b for a in "abcd" for b in "abcdefghijklmnopqrstuvwxyz"][:parts]
    return tuple(f"{benchmark}/h5_{emb}/{folder}.tar.gz.part-{n}" for n in names)


def pinned_archives() -> list[Archive]:
    with open(PINS_TSV, newline="") as f:
        return [
            Archive(r["embodiment"], r["benchmark"], r["folder"], int(r["parts"]),
                    int(r["episodes"]), r["task"])
            for r in csv.DictReader(f, delimiter="\t")
        ]  # fmt: skip


TASKS = {(a.embodiment, a.benchmark, a.folder): a.task for a in pinned_archives() if a.task}
"""(embodiment, benchmark dir, folder) → the trained task text, where the text rule gives
another one."""


@dataclass(frozen=True)
class Embodiment:
    cameras: tuple[str, ...]
    state_keys: tuple[str, ...]
    source: str
    shift: int
    bgr: bool
    benchmarks: tuple[str, ...]
    robot_type: str
    idle: bool = False
    """Leave out idle episodes (``IDLE_VAR``)."""


_J = ("joint_position",)
EMBODIMENTS = {
    "franka_1rgb": Embodiment(("camera_top",), _J, "puppet", 1, True, ("1_0",), "franka_1rgb"),
    "franka_3rgb": Embodiment(
        ("camera_top", "camera_left", "camera_right"), _J, "puppet", 1, True, ("1_0", "1_1", "1_2"),
        "franka_3rgb",
    ),
    "franka_fr3_dual": Embodiment(
        ("camera_front", "camera_top", "camera_left", "camera_right"), _J, "puppet", 1, True,
        ("1_1",), "franka_fr3_dual",
    ),
    "ur_1rgb": Embodiment(("camera_top",), _J, "puppet", 1, True, ("1_0", "1_1"), "ur_1rgb", idle=True),
    "tienkung_gello_1rgb": Embodiment(
        ("camera_top",), _J, "master", 0, False, ("1_0", "1_1"), "tienkung_gello_1rgb"
    ),
    "tienkung_prod1_gello_1rgb": Embodiment(
        ("camera_top",), _J, "master", 0, False, ("1_1",), "tienkung_prod1_gello_1rgb"
    ),
    "tienkung_xsens_1rgb": Embodiment(
        ("camera_top",), ("joint_position", "end_effector"), "puppet", 1, False, ("1_0", "1_1"),
        "tienkung_xsens_1rgb",
    ),
    "agilex_3rgb": Embodiment(
        ("camera_front", "camera_left_wrist", "camera_right_wrist"),
        ("joint_position_left", "joint_position_right"), "master", 0, False, ("1_0", "1_1"),
        "agilex_3rgb",
    ),
}  # fmt: skip
"""Per embodiment: cameras, state/action keys, action source and shift, BGR JPEGs, the
benchmarks that hold it, ``info.json`` robot type."""
assert set(EMBODIMENTS) == set(ROBOMIND)


def source(emb: str, files: tuple[str, ...] | None = None) -> HfSnapshot:
    """The pinned archive parts of ``emb`` (``files``: these only) and the static metadata."""
    if files is None:
        files = tuple(f for a in pinned_archives() if a.embodiment == emb for f in a.files())
    return HfSnapshot(REPO, REVISION, (*files, INSTRUCTIONS, ANNOTATIONS))


def archives(up: Path, src: HfSnapshot, emb: str) -> list[tuple[str, str, list[Path]]]:
    """``(benchmark dir, task folder, parts)`` of every archive part of ``emb`` that ``src``
    pins, in (benchmark, task) order."""
    from fnmatch import fnmatch

    assert src.allow is not None
    out: dict[tuple[str, str], list[Path]] = {}
    for bm in EMBODIMENTS[emb].benchmarks:
        folder = f"benchmark{bm}_compressed/h5_{emb}"
        for part in sorted((up / folder).glob("*.tar.gz.part-*")):
            if not any(fnmatch(f"{folder}/{part.name}", p) for p in src.allow):
                continue
            task = part.name.rsplit(".tar.gz.part-", 1)[0]
            out.setdefault((f"benchmark{bm}_compressed", task), []).append(part)
    return [(bm, task, parts) for (bm, task), parts in sorted(out.items())]


def extract(parts: list[Path], out: Path, whole: bool) -> None:
    """Extract the archive ``parts`` into ``out``. ``whole``: every part is there and the stream
    must end cleanly; else (``--subset``: the first parts only) the members the stream holds
    whole are extracted, the cut one is removed."""
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    member = None
    try:
        with tarfile.open(fileobj=ConcatFile(parts), mode="r|gz") as tf:
            for member in tf:
                tf.extract(member, out, filter="data")
            member = None
    except (EOFError, tarfile.ReadError):
        if whole:
            raise
        if member is not None and (out / member.name).is_file():
            (out / member.name).unlink()


def decode(raw: np.ndarray, bgr: bool) -> np.ndarray | None:
    """One stored frame (JPEG bytes, or raw 720p / 480p RGB) as RGB; None: not decodable."""
    import cv2

    img = cv2.imdecode(np.asarray(raw), cv2.IMREAD_COLOR)
    if img is None:
        flat = np.frombuffer(np.asarray(raw), dtype=np.uint8)
        for h, w in ((720, 1280), (480, 640)):
            if flat.size == h * w * 3:
                img = flat.reshape(h, w, 3)
                break
        else:
            return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if bgr else img


def read_trajectory(
    args: tuple[Path, str, Path],
) -> tuple[np.ndarray, np.ndarray, dict[str, Path]] | str:
    """State, action and the encoded camera mp4s (into ``work``) of one trajectory, or why it
    is left out."""
    import h5py  # type: ignore[import-not-found]

    path, emb, work = args
    spec = EMBODIMENTS[emb]
    with h5py.File(path, "r") as f:
        state = np.concatenate(
            [np.asarray(f[f"puppet/{k}"], dtype=np.float32) for k in spec.state_keys], axis=1
        )
        action = np.concatenate(
            [np.asarray(f[f"{spec.source}/{k}"], dtype=np.float32) for k in spec.state_keys],
            axis=1,
        )
        n = min(len(state), len(action))
        if n < MIN_FRAMES:
            return f"{n} frames"
        state, action = state[:n], action[:n]
        if spec.shift:
            state, action, n = state[: -spec.shift], action[spec.shift :], n - spec.shift
        images = {cam: f[f"observations/rgb_images/{cam}"] for cam in spec.cameras}
        n = min(n, *(len(v) for v in images.values()))
        state, action = state[:n], action[:n]
        if spec.idle and state.var(axis=0).sum() < IDLE_VAR:
            return "idle"
        videos: dict[str, Path] = {}
        for cam, frames in images.items():
            mp4 = work / f"{cam}.mp4"
            encoder = None
            for i in range(n):
                img = decode(frames[i], spec.bgr)
                if img is None or (encoder is not None and img.shape != (*encoder.size, 3)):
                    if encoder is not None:
                        encoder.kill()
                    return f"{cam} frame {i} {'undecodable' if img is None else 'changes shape'}"
                if encoder is None:
                    encoder = FrameEncoder(mp4, FPS, img.shape[0], img.shape[1], X264)
                encoder.write(img)
            assert encoder is not None
            encoder.close()
            videos[f"observation.images.{cam}"] = mp4
    return state, action, videos


def _instructions(up: Path) -> dict[str, str]:
    with open(up / INSTRUCTIONS, newline="") as f:
        return {
            row["task"].strip(): row["instruction"].strip()
            for row in csv.DictReader(f)
            if row.get("task", "").strip() and row.get("instruction", "").strip()
        }


def _annotations(up: Path) -> dict[str, str]:
    """Trajectory dir (``<folder>/success_episodes/<split>/<id>/data``, as in the archive) →
    ``task_summary`` of the AgileX annotations (their ``id`` is that dir under
    ``h5_agilex_3rgb/``)."""
    out = {}
    for entry in json.loads((up / ANNOTATIONS).read_text()):
        key = entry["id"].removeprefix("h5_agilex_3rgb/")
        if key in out:
            raise ValueError(f"{ANNOTATIONS}: two annotations for {key}")
        out[key] = entry["response"]["task_summary"]
    return out


@dataclass(frozen=True)
class RoboMind(Reader):
    emb: str

    def leaves(self, name: str, root: Path, pin: Pin) -> list[Path]:
        return [root_dir(name, root) / f"robomind_{self.emb}_lerobot_v30"]

    def episodes(self, name: str, root: Path, pin: Pin) -> Iterator[v30.Episode]:
        from ..catalog import pins

        (src,) = pin.sources
        assert isinstance(src, HfSnapshot)
        up = upstream_dir(root, src)
        whole = pin == pins()[name]
        rule = text_rules(name).task
        instructions = _instructions(up)
        annotations = _annotations(up) if self.emb == "agilex_3rgb" else {}
        work = root / ".work" / "robomind" / self.emb
        workers = len(os.sched_getaffinity(0))
        items = archives(up, src, self.emb)
        index = trajectory = 0
        with (
            ProcessPoolExecutor(workers) as pool,
            ProcessPoolExecutor(EXTRACT_AHEAD) as unpack,
        ):
            extracted = {}

            def prefetch(j: int) -> None:
                if j < len(items):
                    out = work / f"archive-{j:04d}"
                    extracted[j] = unpack.submit(extract, items[j][2], out, whole)

            for j in range(EXTRACT_AHEAD):
                prefetch(j)
            for j, (bm, task, parts) in enumerate(items):
                logger.info("%s: %s/%s (%d parts)", name, bm, task, len(parts))
                extracted.pop(j).result()
                prefetch(j + EXTRACT_AHEAD)
                archive = work / f"archive-{j:04d}"
                trajectories = sorted(archive.rglob("trajectory.hdf5"))
                text = TASKS.get((self.emb, bm, task)) or rule(instructions.get(task) or task)
                jobs = [
                    (t, self.emb, work / "episodes" / f"{trajectory + i:06d}")
                    for i, t in enumerate(trajectories)
                ]
                trajectory += len(jobs)
                for _, _, d in jobs:
                    d.mkdir(parents=True)
                for (path, _, _), out in zip(jobs, pool.map(read_trajectory, jobs)):
                    rel = path.relative_to(archive)
                    if isinstance(out, str):
                        logger.info("%s: %s left out (%s)", name, rel, out)
                        continue
                    task_text: str | None = text
                    if task in BROKEN_TASKS:
                        summary = annotations.get(rel.parent.as_posix())
                        task_text = robomind_annotation(summary) if summary else None
                    if task_text is None:
                        logger.info("%s: %s left out (no annotation)", name, rel)
                        continue
                    state, action, videos = out
                    yield v30.Episode(index, state, action, task_text, video_src=videos)
                    index += 1
                shutil.rmtree(archive)

    def build(self, name: str, root: Path, pin: Pin, shard: tuple[int, int]) -> None:
        if shard != (0, 1):
            raise ValueError(f"{name}: one leaf, no shards")
        (out,) = self.leaves(name, root, pin)
        work = root / ".work" / "robomind" / self.emb
        if work.exists():
            shutil.rmtree(work)
        v30.write_leaf(self.episodes(name, root, pin), out, FPS, EMBODIMENTS[self.emb].robot_type)
        shutil.rmtree(work)
        info = json.loads((out / "meta" / "info.json").read_text())
        if (info["total_episodes"], info["total_frames"]) != (pin.episodes, pin.frames):
            raise ValueError(
                f"{name}: {info['total_episodes']} episodes / {info['total_frames']} frames, "
                f"pinned {pin.episodes} / {pin.frames}"
            )


TOTALS = {
    "franka_1rgb": (5333, 938509),
    "franka_3rgb": (18665, 2821626),
    "ur_1rgb": (21669, 3231333),
    "franka_fr3_dual": (1774, 374033),
    "agilex_3rgb": (8846, 5673784),
    "tienkung_gello_1rgb": (6081, 3822216),
    "tienkung_prod1_gello_1rgb": (1509, 862663),
    "tienkung_xsens_1rgb": (6125, 3553688),
}
"""Per embodiment: episodes, frames."""


@dataclass(frozen=True)
class _Subset:
    archives: tuple[str, ...]
    """Archive part files (the first parts of an archive: the trajectories they hold whole)."""
    episodes: int
    frames: int


def _parts(bm: str, emb: str, folder: str, parts: int) -> tuple[str, ...]:
    return part_files(f"benchmark{bm}_compressed", emb, folder, parts)


SUBSETS: dict[str, _Subset] = {
    "ur_1rgb": _Subset(_parts("1_0", "ur_1rgb", "green_pepper_in_basket_1", 1), 12, 3486),
    "tienkung_xsens_1rgb": _Subset(
        _parts("1_0", "tienkung_xsens_1rgb", "plug_extract_from", 1)
        + _parts("1_0", "tienkung_xsens_1rgb", "plug_pullout", 1)
        + _parts("1_0", "tienkung_xsens_1rgb", "push_bewak_pick_machine_place_plate", 2),
        186, 93080,
    ),
    "agilex_3rgb": _Subset(
        _parts("1_0", "agilex_3rgb", "1_potatooven_2", 1)
        + _parts("1_0", "agilex_3rgb", "35_putcarrot", 2)
        + _parts("1_0", "agilex_3rgb", "6_applebowloven_2", 1)
        + _parts("1_1", "agilex_3rgb", "6_applebowloven_2", 1),
        166, 95611,
    ),
    "franka_3rgb": _Subset(_parts("1_0", "franka_3rgb", "slide_close_drawer", 1), 29, 3271),
    "tienkung_gello_1rgb": _Subset(
        _parts("1_0", "tienkung_gello_1rgb", "place_bread_table_1128", 1), 103, 63552,
    ),
    "tienkung_prod1_gello_1rgb": _Subset(
        _parts("1_1", "tienkung_prod1_gello_1rgb", "clean_table", 1), 38, 21249
    ),
    "franka_1rgb": _Subset(_parts("1_0", "franka_1rgb", "bread_on_table", 1), 34, 6101),
    "franka_fr3_dual": _Subset(
        _parts("1_1", "franka_fr3_dual", "right_slide_close_drawer", 1), 96, 18678
    ),
}  # fmt: skip
"""Per embodiment, the ``--subset`` slice (whole archives, or the first part of a large one):
UR frozen-arm drops; xsens different-text duplicate groups across two archives (52 removed)
and a ``frozen`` drop; AgileX a same-text duplicate archive in both benchmarks (51 removed) and
a one-word folder (per-episode annotations, unannotated episodes left out); franka_3rgb a short
trajectory; one archive for gello; the first part of one archive for prod1, franka_1rgb and
fr3_dual."""


def _pin(emb: str) -> Pin:
    episodes, frames = TOTALS[emb]
    if episodes != sum(a.episodes for a in pinned_archives() if a.embodiment == emb):
        raise ValueError(f"{PINS_TSV.name}: {emb} episodes differ from the pinned total")
    sub = SUBSETS.get(emb)
    subset = None
    if sub is not None:
        subset = Pin((source(emb, sub.archives),), sub.episodes, sub.frames)
    return Pin((source(emb),), episodes, frames, subset=subset)


PINS = {f"robomind_{e}": _pin(e) for e in ROBOMIND}

RECIPES = {
    f"robomind_{e}": Recipe(
        RoboMind(e),
        curate=("frozen", "ur_frozen_arm") if e == "ur_1rgb" else ("frozen",),
        stats=(op("kept_qk"),),
    )
    for e in ROBOMIND
}
