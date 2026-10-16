"""The deduplicated video-existence check returns lerobot's verdict on every layout."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace

import datasets
import pytest
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

from reki.data.robotics import lerobot_patches as patch

patch.install_lerobot_patches()

pytestmark = pytest.mark.cpu

VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
KEYS = ("observation.images.top", "observation.images.wrist")
FILE_OF_EP = {KEYS[0]: [0, 0, 0, 1, 1, 2, 2], KEYS[1]: [0, 0, 1, 1, 1, 1, 2]}
CHUNK_2 = [0, 0, 0, 1, 1, 1, 1]
FILE_2 = [0, 0, 1, 0, 0, 1, 1]


def _reader(
    root: Path,
    *,
    video_keys: tuple[str, ...] = KEYS,
    hf_eps: Iterable[int] = range(7),
    episodes: list[int] | None = None,
    n_meta_eps: int = 7,
    two_chunks: bool = False,
) -> SimpleNamespace:
    cols: dict[str, list[int]] = {"episode_index": list(range(n_meta_eps))}
    for k in video_keys:
        cols[f"videos/{k}/chunk_index"] = (CHUNK_2 if two_chunks else [0] * 7)[:n_meta_eps]
        cols[f"videos/{k}/file_index"] = (FILE_2 if two_chunks else FILE_OF_EP[k])[:n_meta_eps]
    meta = object.__new__(LeRobotDatasetMetadata)
    meta.root = root
    meta.info = {
        "video_path": VIDEO_PATH,
        "total_episodes": 7,
        "features": {k: {"dtype": "video"} for k in video_keys},
    }
    meta.episodes = datasets.Dataset.from_dict(cols)
    hf = datasets.Dataset.from_dict({"episode_index": [e for e in hf_eps for _ in range(3)]})
    return SimpleNamespace(hf_dataset=hf, episodes=episodes, _meta=meta, _root=root)


def _touch_all(root: Path, skip: tuple[str, int] | None = None, two_chunks: bool = False) -> None:
    """Create every video file; `skip` = (key, file) or, with `two_chunks`, (key, chunk)."""
    for k in KEYS:
        if two_chunks:
            layout = set(zip(CHUNK_2, FILE_2))
            files = [(c, f) for c, f in layout if (k, c) != skip or f != 0]
        else:
            files = [(0, f) for f in set(FILE_OF_EP[k]) if (k, f) != skip]
        for c, f in files:
            p = root / VIDEO_PATH.format(video_key=k, chunk_index=c, file_index=f)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()


def _both(reader: SimpleNamespace) -> tuple[bool, bool]:
    return patch.original_video_check(reader), patch.dedup_video_check(reader)


def test_all_present(tmp_path: Path) -> None:
    _touch_all(tmp_path)
    assert _both(_reader(tmp_path)) == (True, True)


@pytest.mark.parametrize("skip", [(KEYS[0], 0), (KEYS[0], 2), (KEYS[1], 1)])
def test_one_file_missing(tmp_path: Path, skip: tuple[str, int]) -> None:
    _touch_all(tmp_path, skip=skip)
    assert _both(_reader(tmp_path)) == (False, False)


def test_missing_file_only_for_unrequested_episodes(tmp_path: Path) -> None:
    _touch_all(tmp_path, skip=(KEYS[0], 2))
    assert _both(_reader(tmp_path, episodes=[0, 1, 3])) == (True, True)
    assert _both(_reader(tmp_path, episodes=[0, 6])) == (False, False)


def test_episode_out_of_range_raises(tmp_path: Path) -> None:
    _touch_all(tmp_path)
    for fn in (patch.original_video_check, patch.dedup_video_check):
        with pytest.raises(IndexError):
            fn(_reader(tmp_path, n_meta_eps=5))


def test_two_chunks_missing_file_shares_index_with_other_chunk(tmp_path: Path) -> None:
    _touch_all(tmp_path, skip=(KEYS[1], 1), two_chunks=True)
    assert _both(_reader(tmp_path, two_chunks=True)) == (False, False)
