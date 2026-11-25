"""Reader for EgoDex (Apple, CC BY-NC-ND): ARKit hand + camera poses, MANO-fitted fingers.

Upstream: the zips ``part1..5`` and ``extra`` on Apple's CDN (``test`` is not used), pinned by
ETag. Each task folder ``<part>/<task>/`` holds ``<i>.hdf5`` (world SE(3) per joint in
``transforms/``, the device in ``transforms/camera``, the text in the attributes) and ``<i>.mp4``.

Build:

1. MANO fit (GPU; shard ``i/n``: the folders ``[i::n]``), one per task folder: the 21
   otaheri-MANO keypoints of all the folder's episodes are fitted together →
   ``egocentric_raw/egodex_fits/<part>/<task>.npz`` with the finger pose, the fit error and the
   hdf5 wrist/camera poses and text. A frame whose fit error is >= ``FIT_ERR`` m counts as
   untracked for that hand.
2. Unsharded build only: episodes in the order of ``"<part>/<task>__<i>.npz"`` (string sort) go
   round-robin into 96 leaves ``egodex_NN``; the source mp4 is stream-copied. The task text
   loses its scene clauses (``strip_scene``).

Stored values (``egocentric`` module doc): action = world wrist + eigengrasps,
state[0:6] = world camera pose. The MANO model files need a registration at
mano.is.tue.mpg.de: put ``mano_v1_2.zip`` into ``<root>/upstream/mano/``.
"""

from __future__ import annotations

import io
import logging
import multiprocessing
import os
import re
import zipfile
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from ..common import v30
from ..common.download import RangeFile, http, upstream_dir
from ..common.video import stream_frames
from ..pins import HttpFiles, Pin
from ..recipe import Recipe, collection_dir
from . import egocentric as ego

logger = logging.getLogger(__name__)

CDN = "https://ml-site.cdn-apple.com/datasets/egodex"
PARTS = ("extra", "part1", "part2", "part3", "part4", "part5")
SOURCE = HttpFiles(
    "egodex",
    {
        f"{CDN}/extra.zip": "d7e49ce64982f3fa12ecf6e8f0d3e529-3337",
        f"{CDN}/part1.zip": "07290dffadc282f4d01bf78ab5f703db-4793",
        f"{CDN}/part2.zip": "1bb5321e52a3204c791561dd30204968-4877",
        f"{CDN}/part3.zip": "06bc7804d11c83b4c899d1d6a7948dd3-4862",
        f"{CDN}/part4.zip": "91b783577e3b4e3e7e6d8aa712b905e8-4908",
        f"{CDN}/part5.zip": "7ee9bb3f5aa44ec61126678b419d814c-4937",
    },
)


@dataclass(frozen=True)
class EgodexPin(Pin):
    """``folders``: the ``(part, task)`` folders read (range reads of the zips); None: every
    folder of the downloaded zips."""

    folders: tuple[tuple[str, str], ...] | None = None
    files: tuple[str, ...] = ()
    """``<part>/<task>/<stem>`` episodes fetched alone (``.hdf5`` and ``.mp4``); a folder with
    none listed is fetched whole."""


PIN = EgodexPin(
    (SOURCE,),
    episodes=334_991,
    frames=88_753_873,
    subset=EgodexPin(
        (SOURCE,),
        episodes=9,
        frames=4997,
        folders=(("part1", "fry_bread"), ("part4", "play_reset_connect_four")),
        files=("part4/play_reset_connect_four/236", "part4/play_reset_connect_four/1125"),
    ),
)
"""The ``--subset``: ``part1/fry_bread`` (7 episodes) and one same-folder duplicate pair of
``part4/play_reset_connect_four``; the dropped copy is alone in its leaf, which is left empty."""
MANO_SHA256 = {
    "MANO_LEFT.pkl": "c4022f7083f2ca7c78b2b3d595abbab52debd32b09d372b16923a801f0ea6a30",
    "MANO_RIGHT.pkl": "45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767",
}
FPS = 30
FIT_ERR = 0.05
SIDES = ("left", "right")
WRITE_WORKERS = 16
"""Leaves written at once by the unsharded ``build`` (each leaf is independent)."""


def _keypoint_names(side: str) -> list[str]:
    """EgoDex joints in otaheri-MANO 21-keypoint order (0 wrist, 1-3 index, 4-6 middle,
    7-9 little, 10-12 ring, 13-15 thumb, 16-20 tips thumb/index/middle/ring/little).
    Knuckle → MCP, IntermediateBase → PIP, IntermediateTip → DIP; metacarpals unused."""
    finger = [
        f"{side}{f}Finger{p}"
        for f in ("Index", "Middle", "Little", "Ring")
        for p in ("Knuckle", "IntermediateBase", "IntermediateTip")
    ]
    thumb = [f"{side}Thumb{p}" for p in ("Knuckle", "IntermediateBase", "IntermediateTip", "Tip")]
    tips = [f"{side}{f}FingerTip" for f in ("Index", "Middle", "Ring", "Little")]
    return [f"{side}Hand", *finger, *thumb, *tips]


def _text(attrs: Any, task: str) -> str:
    """``llm_description``, else ``description``, else the folder name (``insert_remove_bookshelf``
    → ``"Insert remove bookshelf."``). Upstream writes ``"Description unavailable."`` for no text."""
    for key in ("llm_description", "description"):
        value = str(attrs.get(key) or "").strip()
        if value and value not in ("None", "Description unavailable."):
            return value
    return task.replace("_", " ").capitalize() + "."


_ADJ = r"(?:(?!(?:while|with|and|on|at|against|using|then)\b)[\w\-]+ ){0,4}?"
_SURFACE = r"(?:tablecloth|table|desk|counter|surface)\b"
_BACKGROUND = rf"(?:(?:a|an|the) )?{_ADJ}background\b"
_SCENE = [
    re.compile(p, re.I)
    for p in (
        r"^While (?:sitting|standing|seated), ",
        # ", while sitting at a wooden table with a red background"
        rf",? while (?:sitting|standing|seated)(?: (?:at|on|in front of|by|near|against) "
        rf"(?:(?:a|an|the) {_ADJ}{_SURFACE}|{_BACKGROUND}))?(?: (?:covered )?with (?:(?:a|an|the) )?"
        rf"{_ADJ}(?:tablecloth\b|background\b)(?: and {_BACKGROUND})?)?",
        # "on a blue tablecloth with a pink background"
        rf",? (?:on|at|over|onto|across) (?:a|an|the) {_ADJ}tablecloth\b"
        rf"(?: (?:with|and|against) {_BACKGROUND})?",
        # "on a wooden table with a red background"
        rf",? (?:on|at) (?:a|an|the) {_ADJ}(?:table|surface|desk)\b (?:covered )?with "
        rf"(?:a|an|the) {_ADJ}(?:tablecloth\b|background\b)(?: and {_BACKGROUND})?",
        rf",? (?:with|against|on|in front of|and) {_BACKGROUND}",
        r",? (?:without (?:a )?|with no |and no )tablecloth\b",
        r",? (?:with the person performing the task )?in a sitting position",
        r",? (?:while )?(?:positioned )?(?:sitting|standing)(?=\.?$)",
    )
]


def strip_scene(text: str) -> str:
    """``text`` without its scene clauses (``"Stack five bowls on a green tablecloth while sitting
    against a blue background."`` → ``"Stack five bowls."``): the camera shows the scene, and no
    robot prompt describes it."""
    for pattern in _SCENE:
        text = pattern.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.])", r"\1", text)
    text = re.sub(r",\s*\.", ".", text)
    text = re.sub(r"(?:,| and| placed| positioned| located)+\s*$", "", text.rstrip(".")).strip()
    return text[0].upper() + text[1:] + "."


class Archive:
    """The members of the downloaded parts: ``<part>.zip`` (full download) or the extracted
    members under ``<part>/`` (``--subset``)."""

    def __init__(self, src: Path) -> None:
        self.src = src
        self.zips: dict[str, zipfile.ZipFile] = {}

    def _zip(self, part: str) -> zipfile.ZipFile | None:
        path = self.src / f"{part}.zip"
        if part not in self.zips and path.exists():
            self.zips[part] = zipfile.ZipFile(path)
        return self.zips.get(part)

    def folders(self, part: str) -> dict[str, list[str]]:
        """``task → sorted hdf5 members`` of one part."""
        zf = self._zip(part)
        if zf is not None:
            names = zf.namelist()
        elif (self.src / part).is_dir():
            names = [p.relative_to(self.src).as_posix() for p in (self.src / part).rglob("*")]
        else:
            raise FileNotFoundError(f"{self.src}: neither {part}.zip nor {part}/")
        by_task: dict[str, list[str]] = defaultdict(list)
        for m in names:
            if m.endswith(".hdf5"):
                by_task[m.split("/")[-2]].append(m)
        return {task: sorted(members) for task, members in sorted(by_task.items())}

    def read(self, member: str) -> bytes:
        zf = self._zip(member.split("/", 1)[0])
        return zf.read(member) if zf is not None else (self.src / member).read_bytes()


# ------------------------------------------------------------------ MANO fits


def mano_dir(root: Path) -> Path:
    """The checked MANO model folder (extracted from ``upstream/mano/mano_v1_2.zip``)."""
    import hashlib

    base = root / "upstream" / "mano"
    models = base / "mano_v1_2" / "models"
    if not models.is_dir():
        archive = base / "mano_v1_2.zip"
        if not archive.exists():
            raise FileNotFoundError(
                f"{archive}: register at https://mano.is.tue.mpg.de, download mano_v1_2.zip "
                "(Models & Code) and put it there"
            )
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(base)
    for name, sha in MANO_SHA256.items():
        if hashlib.sha256((models / name).read_bytes()).hexdigest() != sha:
            raise ValueError(f"{models / name}: sha256 differs from the pinned MANO v1.2 file")
    return models


def fit_folder(archive: Archive, members: list[str], out: Path, models: Path) -> None:
    """MANO-fit all episodes of one task folder together → ``out`` (written atomically)."""
    import h5py  # type: ignore[import-not-found]

    from .mano_fit import fit_hand

    cols: dict[str, list[np.ndarray]] = defaultdict(list)
    lengths, texts = [], []
    for member in members:
        with h5py.File(io.BytesIO(archive.read(member)), "r") as h:
            tr = h["transforms"]
            n = tr["rightHand"].shape[0]
            for side in SIDES:
                cols[f"{side}_kp"].append(
                    np.stack(
                        [np.asarray(tr[k][:n])[:, :3, 3] for k in _keypoint_names(side)], 1
                    ).astype(np.float32)
                )
                wrist = np.asarray(tr[f"{side}Hand"][:n])
                cols[f"{side}_wrist_t"].append(wrist[:, :3, 3].astype(np.float32))
                cols[f"{side}_wrist_aa"].append(
                    ego.rotmat_to_rotvec(wrist[:, :3, :3]).astype(np.float32)
                )
            camera = tr["camera"][:]
            cols["camera_t"].append(camera[:, :3, 3].astype(np.float32))
            cols["camera_aa"].append(
                ego.rotmat_to_rotvec(camera[:, :3, :3].astype(np.float64)).astype(np.float32)
            )
            if len(camera) != n:
                raise ValueError(f"{member}: {len(camera)} camera poses, {n} hand poses")
            lengths.append(n)
            texts.append(_text(h.attrs, member.split("/")[-2]))
    arrays: dict[str, Any] = {k: np.concatenate(v) for k, v in cols.items()}
    for side, is_right in zip(SIDES, (False, True)):
        fit = fit_hand(arrays.pop(f"{side}_kp"), lengths, is_right, models)
        arrays[f"{side}_finger_aa"] = fit["finger_aa"]
        arrays[f"{side}_fit_err"] = fit["fit_err"]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(
        tmp,
        members=np.array(members),
        lengths=np.array(lengths),
        texts=np.array(texts),
        **arrays,
    )
    tmp.rename(out)


# ------------------------------------------------------------------ episodes


def episode_arrays(fits: dict, k: int, pca: ego.Eigengrasps) -> tuple[np.ndarray, np.ndarray]:
    """``(state [T, 12], action [T, 32])`` of the ``k``-th episode of a folder's fits."""
    start = int(np.sum(fits["lengths"][:k]))
    sl = slice(start, start + int(fits["lengths"][k]))
    action = np.concatenate(
        [
            ego.hand_columns(
                fits[f"{side}_wrist_t"][sl],
                fits[f"{side}_wrist_aa"][sl],
                pca.transform(fits[f"{side}_finger_aa"][sl].astype(np.float64)),
                fits[f"{side}_fit_err"][sl] < FIT_ERR,
            )
            for side in SIDES
        ],
        axis=1,
    )
    return ego.camera_state(fits["camera_t"][sl], fits["camera_aa"][sl]), action


@dataclass(frozen=True)
class _Ref:
    """One episode: its fits file and its position in the folder."""

    fits: Path
    k: int


class _FitsCache:
    def __init__(self) -> None:
        self.path: Path | None = None
        self.data: dict = {}

    def get(self, path: Path) -> dict:
        if path != self.path:
            with np.load(path) as d:
                self.data = {k: d[k] for k in d.files}
            self.path = path
        return self.data


def _episode_order(folders: list[tuple[str, str, list[str]]]) -> list[tuple[str, str, int]]:
    """``(part, task, position in the folder)`` of every episode, in build order: the string
    order of ``"<part>/<task>__<i>.npz"``."""
    refs = {
        f"{part}/{task}__{member.rsplit('/', 1)[-1][: -len('.hdf5')]}.npz": (part, task, k)
        for part, task, members in folders
        for k, member in enumerate(members)
    }
    return [refs[key] for key in sorted(refs)]


class Egodex(ego.EgoReader):
    prefix = "egodex"

    def __init__(self) -> None:
        self._labels: dict[tuple[Path, int], list[str]] = {}

    def download(self, name: str, root: Path, pin: Pin) -> None:
        assert isinstance(pin, EgodexPin)
        mano_dir(root)
        (source,) = pin.sources
        assert isinstance(source, HttpFiles)
        out = upstream_dir(root, source)
        if pin.folders is None:
            http(source, out)
            return
        for part, task in pin.folders:
            url = f"{CDN}/{part}.zip"
            with zipfile.ZipFile(RangeFile(url, etag=source.etags[url])) as zf:
                members = [m for m in zf.namelist() if m.startswith(f"{part}/{task}/")]
                stems = [f for f in pin.files if f.startswith(f"{part}/{task}/")]
                if stems:
                    members = [m for m in members if m.rsplit(".", 1)[0] in stems]
                for m in members:
                    dst = out / m
                    if m.endswith("/") or dst.exists():
                        continue
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dst.with_name(dst.name + ".part")
                    tmp.write_bytes(zf.read(m))
                    tmp.rename(dst)
            logger.info("%s: %s/%s, %d members", name, part, task, len(members))

    def _folders(self, root: Path, pin: Pin) -> tuple[Archive, list[tuple[str, str, list[str]]]]:
        """The archive and its ``(part, task, members)`` folders that ``pin`` reads."""
        assert isinstance(pin, EgodexPin)
        (source,) = pin.sources
        archive = Archive(upstream_dir(root, source))
        if pin.folders is None:
            return archive, [(p, t, m) for p in PARTS for t, m in archive.folders(p).items()]
        return archive, [(p, t, archive.folders(p)[t]) for p, t in pin.folders]

    def leaves(self, name: str, root: Path, pin: Pin) -> list[Path]:
        """The pinned leaves (a ``build`` shard writes none)."""
        return [self.leaf_dir(name, root, i) for i in range(min(ego.LEAVES, pin.episodes))]

    def dedup_labels(self, name: str, root: Path, pin: Pin, leaf: Path) -> list[str] | None:
        """The upstream task folder ``<part>/<task>`` of each episode: in-folder copies differ
        only by LLM paraphrases; the list of excluded episodes keeps the copy of the right
        folder."""
        key = (root, id(pin))
        if key not in self._labels:
            self._labels[key] = [
                f"{p}/{t}" for p, t, _ in _episode_order(self._folders(root, pin)[1])
            ]
        return self._labels[key][int(leaf.name.rsplit("_", 1)[1]) :: ego.LEAVES]

    finish_after_shards = True

    def build(self, name: str, root: Path, pin: Pin, shard: tuple[int, int]) -> None:
        """Shard ``i/n`` MANO-fits the folders ``[i::n]`` (GPU); the unsharded build fits the
        folders still missing, then writes every leaf, ``WRITE_WORKERS`` at a time."""
        archive, folders = self._folders(root, pin)
        fits_dir = collection_dir(name, root).parent / "egodex_fits"
        i, n = shard
        models: Path | None = None
        for part, task, members in folders[i::n]:
            path = fits_dir / part / f"{task}.npz"
            if not path.exists():
                models = models or mano_dir(root)
                fit_folder(archive, members, path, models)
                logger.info("%s: fitted %s/%s (%d episodes)", name, part, task, len(members))
        if shard != (0, 1):
            return
        order = [_Ref(fits_dir / p / f"{t}.npz", k) for p, t, k in _episode_order(folders)]
        if len(order) != pin.episodes:
            raise ValueError(f"{name}: {len(order)} episodes, pinned {pin.episodes}")
        numbered = list(enumerate(order))
        leaves = range(min(ego.LEAVES, len(order)))
        refs = [numbered[leaf :: ego.LEAVES] for leaf in leaves]
        workers = min(WRITE_WORKERS, len(os.sched_getaffinity(0)), len(refs))
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(workers, mp_context=ctx) as pool:
            write = partial(_write_leaf, self, name, root, archive.src)
            for done, leaf in enumerate(pool.map(write, leaves, refs), 1):
                logger.info("%s: wrote %s_%02d (%d / %d)", name, name, leaf, done, len(refs))


def _write_leaf(
    reader: Egodex, name: str, root: Path, src: Path, leaf: int, refs: list[tuple[int, _Ref]]
) -> int:
    """Leaf ``leaf`` from its episodes ``(g, ref)`` (a process-pool task)."""
    archive = Archive(src)
    pca = ego.Eigengrasps.load()
    cache = _FitsCache()

    def episodes(work: Path) -> Iterator[v30.Episode]:
        for g, ref in refs:
            fits = cache.get(ref.fits)
            state, action = episode_arrays(fits, ref.k, pca)
            mp4 = work / f"{g:06d}.mp4"
            mp4.write_bytes(archive.read(str(fits["members"][ref.k])[:-5] + ".mp4"))
            if stream_frames(mp4) != len(action):
                raise ValueError(f"{mp4}: {stream_frames(mp4)} frames, {len(action)} poses")
            text = str(fits["texts"][ref.k])
            task = strip_scene(text) if text else ego.FALLBACK_TASK
            yield v30.Episode(g, state, action, task, video_src={ego.VIDEO_KEY: mp4})

    reader.write_leaf(name, root, leaf, episodes, FPS)
    return leaf


PINS = {"egodex": PIN}
RECIPES = {
    "egodex": Recipe(
        Egodex(),
        curate=(),
        stats=(),
        horizon=(
            ego.ego_stats,
            partial(ego.ego_wrist, windows=8, max_episodes=5000, interleave=True),
        ),
    )
}
