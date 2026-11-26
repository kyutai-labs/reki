"""DROID: ``lerobot/droid_1.0.1`` (LeRobot v3.0) at 0eabc778 → one leaf
``open_x_embodiment_lerobot/droid_v30_lerobot``, every upstream episode, values unchanged
(``lerobot_v2x.PassthroughV30``).

The commit holds 70 data shards and 29 videos that its meta does not reference (an upstream
repack left them): ``download`` fetches the meta first, then only the files it references.

Curation: dedup, ``failure`` (episodes the collector labeled as not successful), ``frozen``,
``short_video`` and ``video_cuts`` (video files that stitch several recordings).
No subtask lookup: DROID trains with its task text (registry ``subtask_lookup``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ..common import download
from ..common.stats import op
from ..pins import HfSnapshot, Pin
from ..recipe import Recipe
from .lerobot_v2x import PassthroughV30

NAME = "droid"
REPO = "lerobot/droid_1.0.1"
REVISION = "0eabc778f959c54b8c5aa3626cc1128d2d2e54d4"
CAMERAS = ("exterior_1_left", "exterior_2_left", "wrist_left")

SUBSET = (28118, 28398, 28538, 57587, 92067, 92637)
"""Upstream episodes of ``--subset``: a same-text duplicate pair (92067, 92637), a different-text
pair (28538, 57587), a ``frozen`` and ``failure`` drop (28398) and episode 28118."""
SUBSET_FILES = (
    "meta/**",
    *(f"data/chunk-000/file-{f:03d}.parquet" for f in (25, 51, 82)),
    *(
        f"videos/observation.images.{cam}/chunk-000/file-{f:03d}.mp4"
        for cam, files in (
            ("exterior_1_left", (89, 90, 181, 290, 292)),
            ("exterior_2_left", (88, 89, 179, 287, 288)),
            ("wrist_left", (53, 109, 175, 176)),
        )
        for f in files
    ),
)
SUBSET_FRAMES = 752


def _files(table: pa.Table, path: str, prefix: str, **fmt: str) -> set[str]:
    chunks, files = (
        table.column(f"{prefix}/{k}").to_pylist() for k in ("chunk_index", "file_index")
    )
    return {path.format(chunk_index=c, file_index=f, **fmt) for c, f in zip(chunks, files)}


def referenced_files(src: Path) -> list[str]:
    """The data and video files that ``meta/episodes`` of the v3.0 dataset ``src`` references."""
    info = json.loads((src / "meta" / "info.json").read_text())
    files = sorted((src / "meta" / "episodes").glob("**/*.parquet"))
    table = pa.concat_tables([pq.read_table(f) for f in files])
    out = _files(table, info["data_path"], "data")
    for cam in CAMERAS:
        key = f"observation.images.{cam}"
        out |= _files(table, info["video_path"], f"videos/{key}", video_key=key)
    return sorted(out)


@dataclass(frozen=True)
class Droid(PassthroughV30):
    def download(self, name: str, root: Path, pin: Pin) -> None:
        (src,) = pin.sources
        assert isinstance(src, HfSnapshot)
        out = download.upstream_dir(root, src)
        if src.allow is not None:
            download.fetch(src, out)
            return
        download.fetch(HfSnapshot(REPO, REVISION, ("meta/**",)), out)
        download.fetch(HfSnapshot(REPO, REVISION, ("meta/**", *referenced_files(out))), out)


PINS = {
    NAME: Pin(
        (HfSnapshot(REPO, REVISION),),
        episodes=95658,
        frames=27630375,
        subset=Pin(
            (HfSnapshot(REPO, REVISION, SUBSET_FILES),),
            episodes=len(SUBSET),
            frames=SUBSET_FRAMES,
            select=SUBSET,
        ),
    ),
}

RECIPES = {
    NAME: Recipe(
        Droid("droid_v30_lerobot"),
        curate=("failure", "frozen", "short_video", "video_cuts"),
        stats=(op("kept_qk"),),
    )
}
