"""Reader for EgoVerse (Aria glasses, GaTech-RL2/EgoVerse): MPS hand keypoints, MANO-fitted
hands, the head pose as the camera.

Upstream: a Cloudflare R2 bucket, read through its S3 API with the credentials of the EgoVerse
team (``R2_ENDPOINT_URL``, ``R2_ACCESS_KEY_ID``, ``R2_SECRET_ACCESS_KEY``, ``R2_BUCKET``).
Each episode ``<name>`` (its start time) is the Zarr v3 group ``processed_v3/aria/<name>.zarr``
(``total_frames`` and the text in the attributes; arrays padded past ``total_frames``) and the
video ``processed_v3/aria/<name>.mp4`` (480×640, 30 fps, one frame per row). The bucket has no
versions: ``pins_egoverse.tsv`` lists the 2,514 Aria episodes used, each with its frame count and
the digest of the objects read (key, size, ETag); ``download`` checks it before the transfer
and each file's size (and MD5 for a one-part ETag) after it. Read per episode: the mp4,
``obs_head_pose`` and the ``<side>.obs_aria_keypoints`` of each recorded hand (a one-hand episode
has no array for the other hand).

Build (GPU; shard ``i/n``: the leaves ``[i::n]``): episodes in name order go round-robin into
96 leaves ``egoverse_NN``. Per leaf, each hand of all its episodes is MANO-fitted together
(``mano_fit.fit_hand``): the 20 Aria keypoints that have an otaheri-MANO match (the thumb base
has none). A frame is tracked when its wrist keypoint is finite and the fit error is below
``FIT_ERR``. The upstream mp4 is stream-copied.

Stored values (``egocentric`` module doc): action = the world pose of the MANO wrist joint (fitted
global orient) + eigengrasps; state[0:6] = ``obs_head_pose`` (translation, quaternion **wxyz**).
The MANO model files: as for EgoDex, ``mano_v1_2.zip`` in ``<root>/upstream/mano/``.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from ..common import v30
from ..common.download import CHANGED
from ..common.video import stream_frames
from ..pins import Pin
from ..recipe import Recipe
from . import egocentric as ego
from .egodex import FIT_ERR, mano_dir

logger = logging.getLogger(__name__)

PINS_TSV = Path(__file__).parents[1] / "pins_egoverse.tsv"
PREFIX = "processed_v3/aria/"
ENV = ("R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")
SIDES = ("left", "right")
ARRAYS = ("obs_head_pose", *(f"{side}.obs_aria_keypoints" for side in SIDES))
FPS = 30
OTAHERI_FROM_ARIA = (5, 8, 9, 10, 11, 12, 13, 17, 18, 19, 14, 15, 16, None, 6, 7, 0, 1, 2, 3, 4)
"""Per otaheri-MANO keypoint (0 wrist, 1-3 index, 4-6 middle, 7-9 little, 10-12 ring, 13-15
thumb, 16-20 tips thumb/index/middle/ring/little), its Aria MPS keypoint (0-4 tips thumb … little,
5 wrist, 6-7 thumb, 8-19 index … little, 3 each, 20 palm). The thumb base 13 has none."""
DOWNLOAD_WORKERS = 16


def episode_rows() -> list[tuple[str, int, str]]:
    """``(name, frames, digest)`` of every pinned episode, in name order."""
    with open(PINS_TSV, newline="") as f:
        return [
            (r["episode"], int(r["frames"]), r["digest"]) for r in csv.DictReader(f, delimiter="\t")
        ]


@dataclass(frozen=True)
class EgoversePin(Pin):
    """``names``: the episodes read; ``leaves``: how many leaves they go round-robin into."""

    names: tuple[str, ...] = ()
    leaves: int = ego.LEAVES


_ROWS = episode_rows()
SUBSET = (
    "2025-09-20-17-42-51-000000",
    "2025-11-23-19-49-48-686000",
    "2025-12-17-00-16-23-805000",
    "2026-01-19-22-59-23-000000",
    "2026-04-14-04-03-19-000000",
)
"""``--subset`` episodes, in 2 leaves: two-hand ``fold_clothes``, ``cup_on_saucer`` and
``pick_place`` episodes, a right-hand-only and a left-hand-only one."""
PIN = EgoversePin(
    (),
    episodes=len(_ROWS),
    frames=sum(frames for _, frames, _ in _ROWS),
    names=tuple(name for name, _, _ in _ROWS),
    subset=EgoversePin(
        (),
        episodes=len(SUBSET),
        frames=sum(frames for name, frames, _ in _ROWS if name in SUBSET),
        names=SUBSET,
        leaves=2,
    ),
)


# ------------------------------------------------------------------ download


def wanted(key: str) -> bool:
    """Whether the build reads the object ``key`` (relative to ``PREFIX``)."""
    if key.endswith(".mp4"):
        return True
    _, sep, rest = key.partition(".zarr/")
    return bool(sep) and (rest == "zarr.json" or rest.split("/", 1)[0] in ARRAYS)


def digest(objects: list[tuple[str, int, str]]) -> str:
    """The pinned digest of ``(key, size, ETag)`` objects: sha1 of their sorted TSV lines."""
    lines = "".join(f"{k}\t{s}\t{e}\n" for k, s, e in sorted(objects))
    return hashlib.sha1(lines.encode()).hexdigest()


def _client() -> Any:
    missing = [v for v in ENV if not os.environ.get(v)]
    if missing:
        raise PermissionError(
            f"egoverse: set {', '.join(missing)} (R2 credentials from the EgoVerse team, "
            "docs/pretraining_data.md#access)"
        )
    import boto3  # type: ignore[import-not-found]

    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def listing(client: Any, bucket: str, name: str) -> list[tuple[str, int, str]]:
    """The ``(key, size, ETag)`` objects of episode ``name`` that the build reads."""
    out = []
    for prefix in (f"{PREFIX}{name}.zarr/", f"{PREFIX}{name}.mp4"):
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            for o in page.get("Contents", []):
                key = o["Key"][len(PREFIX) :]
                if wanted(key):
                    out.append((key, int(o["Size"]), o["ETag"].strip('"')))
    return out


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_episode(client: Any, bucket: str, name: str, pinned: str, out: Path) -> None:
    """Episode ``name`` into ``out`` after checking its objects against the ``pinned`` digest;
    a file of the right size is kept (resumable)."""
    objects = listing(client, bucket, name)
    if digest(objects) != pinned:
        raise ValueError(f"egoverse: {name}: the objects differ from the pin {CHANGED}")
    for key, size, etag in objects:
        dst = out / key
        if dst.exists() and dst.stat().st_size == size:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        client.download_file(bucket, PREFIX + key, str(tmp))
        if tmp.stat().st_size != size or ("-" not in etag and _md5(tmp) != etag):
            raise ValueError(f"egoverse: {key}: the download differs from its listing (size, MD5)")
        tmp.rename(dst)


def upstream(root: Path) -> Path:
    return root / "upstream" / "egoverse"


# ------------------------------------------------------------------ episodes


def task_text(attrs: dict) -> str:
    """``task_description``, else ``task_name`` with spaces, capitalized, with a final period."""
    text = str(attrs.get("task_description") or "").strip()
    text = text or str(attrs.get("task_name") or "").replace("_", " ").strip()
    if not text:
        return ego.FALLBACK_TASK
    return text.capitalize() + ("" if text.endswith(".") else ".")


def otaheri_keypoints(aria: np.ndarray) -> np.ndarray:
    """``[T, 21, 3]`` Aria keypoints → otaheri-MANO order; the thumb base and the points of an
    all-zero frame are NaN (missing for the fit)."""
    aria = np.asarray(aria, np.float32)
    out = np.full(aria.shape, np.nan, np.float32)
    for k, a in enumerate(OTAHERI_FROM_ARIA):
        if a is not None:
            out[:, k] = aria[:, a]
    out[~aria.reshape(len(aria), -1).any(axis=1)] = np.nan
    return out


def camera_pose(head: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``obs_head_pose`` ``[T, 7]`` (translation, quaternion wxyz) → translation, axis-angle."""
    head = np.asarray(head, np.float64)
    quat = head[:, 3:7]
    if (np.abs(np.linalg.norm(quat, axis=1) - 1.0) > 1e-3).any():
        raise ValueError("obs_head_pose has a non-unit quaternion")
    return head[:, :3], Rotation.from_quat(quat[:, [1, 2, 3, 0]]).as_rotvec()


@dataclass
class Raw:
    """One episode's upstream arrays, cut at ``total_frames``."""

    name: str
    task: str
    head: np.ndarray
    keypoints: dict[str, np.ndarray]
    """Per recorded hand, ``[T, 21, 3]`` Aria keypoints."""


def read_episode(src: Path, name: str, frames: int) -> Raw:
    import zarr  # type: ignore[import-not-found]

    group = zarr.open_group(src / f"{name}.zarr", mode="r")
    attrs = dict(group.attrs)
    n = int(attrs["total_frames"])
    if n != frames:
        raise ValueError(f"egoverse: {name}: {n} frames, pinned {frames}")
    if float(attrs.get("fps", FPS)) != FPS:
        raise ValueError(f"egoverse: {name}: {attrs['fps']} fps, expected {FPS}")
    keypoints = {
        side: np.asarray(group[f"{side}.obs_aria_keypoints"][:n]).reshape(n, 21, 3)
        for side in SIDES
        if f"{side}.obs_aria_keypoints" in group
    }
    head = np.asarray(group["obs_head_pose"][:n])
    if len(head) != n or any(len(k) != n for k in keypoints.values()):
        raise ValueError(f"egoverse: {name}: an array is shorter than {n} frames")
    return Raw(name, task_text(attrs), head, keypoints)


def fit_hands(raws: list[Raw], models: Path) -> list[dict[str, np.ndarray]]:
    """Per episode, per recorded hand, the MANO fit (``fit_hand`` keys) and ``tracked``; each hand
    of all ``raws`` is fitted at once."""
    from .mano_fit import fit_hand

    out: list[dict[str, np.ndarray]] = [{} for _ in raws]
    for side, is_right in zip(SIDES, (False, True)):
        have = [i for i, r in enumerate(raws) if side in r.keypoints]
        if not have:
            continue
        keypoints = np.concatenate([otaheri_keypoints(raws[i].keypoints[side]) for i in have])
        lengths = [len(raws[i].head) for i in have]
        fit = fit_hand(keypoints, lengths, is_right, models)
        tracked = (np.abs(keypoints[:, 0]) < 1e8).all(axis=1) & (fit["fit_err"] < FIT_ERR)
        start = 0
        for i, n in zip(have, lengths):
            sl = slice(start, start + n)
            out[i] |= {f"{side}_{k}": v[sl] for k, v in fit.items()}
            out[i][f"{side}_tracked"] = tracked[sl]
            start += n
    return out


def episode_arrays(
    raw: Raw, fit: dict[str, np.ndarray], pca: ego.Eigengrasps
) -> tuple[np.ndarray, np.ndarray]:
    """``(state [T, 12], action [T, 32])``; a hand without keypoints is untracked."""
    n = len(raw.head)
    blocks = []
    for side in SIDES:
        if side not in raw.keypoints:
            blocks.append(np.zeros((n, 16), np.float32))
            continue
        blocks.append(
            ego.hand_columns(
                fit[f"{side}_wrist_t"],
                fit[f"{side}_wrist_aa"],
                pca.transform(fit[f"{side}_finger_aa"]),
                fit[f"{side}_tracked"],
            )
        )
    return ego.camera_state(*camera_pose(raw.head)), np.concatenate(blocks, axis=1)


# ------------------------------------------------------------------ reader


class Egoverse(ego.EgoReader):
    prefix = "egoverse"

    def download(self, name: str, root: Path, pin: Pin) -> None:
        assert isinstance(pin, EgoversePin)
        mano_dir(root)
        client = _client()
        bucket = os.environ["R2_BUCKET"]
        digests = {n: d for n, _, d in _ROWS}
        out = upstream(root)

        def one(episode: str) -> None:
            fetch_episode(client, bucket, episode, digests[episode], out)

        with ThreadPoolExecutor(DOWNLOAD_WORKERS) as pool:
            for done, _ in enumerate(pool.map(one, pin.names), 1):
                if done % 100 == 0 or done == len(pin.names):
                    logger.info("%s: %d / %d episodes", name, done, len(pin.names))

    def leaves(self, name: str, root: Path, pin: Pin) -> list[Path]:
        assert isinstance(pin, EgoversePin)
        return [self.leaf_dir(name, root, i) for i in range(min(pin.leaves, pin.episodes))]

    def _episodes(
        self, numbered: list[tuple[int, str]], root: Path, work: Path
    ) -> Iterator[v30.Episode]:
        frames = {n: f for n, f, _ in _ROWS}
        src = upstream(root)
        raws = [read_episode(src, ep, frames[ep]) for _, ep in numbered]
        fits = fit_hands(raws, mano_dir(root))
        pca = ego.Eigengrasps.load()
        for (g, ep), raw, fit in zip(numbered, raws, fits):
            mp4 = src / f"{ep}.mp4"
            if stream_frames(mp4) != len(raw.head):
                raise ValueError(f"{mp4}: {stream_frames(mp4)} frames, {len(raw.head)} poses")
            state, action = episode_arrays(raw, fit, pca)
            yield v30.Episode(g, state, action, raw.task, video_src={ego.VIDEO_KEY: mp4})

    def build(self, name: str, root: Path, pin: Pin, shard: tuple[int, int]) -> None:
        assert isinstance(pin, EgoversePin)
        i, n = shard
        numbered = list(enumerate(pin.names))
        for leaf in range(min(pin.leaves, len(numbered)))[i::n]:
            refs = numbered[leaf :: pin.leaves]
            self.write_leaf(name, root, leaf, partial(self._episodes, refs, root), FPS)


PINS = {"egoverse": PIN}
RECIPES = {
    "egoverse": Recipe(
        Egoverse(),
        curate=(),
        stats=(),
        horizon=(ego.ego_stats, partial(ego.ego_wrist, windows=8)),
    )
}
