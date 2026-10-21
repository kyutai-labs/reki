# Parts adapted from LeRobot (github.com/huggingface/lerobot @ aa9cc9bd43e9eba92a32aaadde2a09c90b5836cf),
# src/lerobot/datasets/video_utils.py and dataset_reader.py: changed to return uint8, bound the
# decoder cache and stat each video file once. Copyright 2024 The HuggingFace Inc. team. Licensed
# under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). See NOTICE.
"""Patches to the pinned LeRobot, installed by ``install_lerobot_patches()`` before a
``LeRobotDataset`` is built (``lerobot_io.load_lerobot_dir``).

1. uint8 video decoders. LeRobot's ``decode_video_frames_*`` cast the decoded uint8 frames to
   ``float32 / 255``; the image pipeline casts them back to uint8 before resizing. The
   replacements return uint8 (the last cast removed). The dispatcher ``decode_video_frames``
   resolves the backend on ``lerobot.datasets.video_utils`` at call time, so rebinding the two
   backends is sufficient. The torchcodec one also bounds LeRobot's decoder cache (an
   unbounded dict) to an LRU that closes evicted file handles.
2. One stat per video file. ``DatasetReader._check_cached_episodes_sufficient`` runs inside
   every ``LeRobotDataset(...)`` and stats one path per episode and video key. v3.0 packs many
   episodes into each mp4, so most of these paths repeat. The replacement stats each video
   file once. It returns the same result and raises the same IndexError, in LeRobot's episode
   order.

``thread_safe_tqdm_lock`` is a third, scoped patch for building leaves in threads.
"""

from __future__ import annotations

import contextlib
import logging
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

_installed = False
original_video_check: Any = None
"""LeRobot's ``_check_cached_episodes_sufficient``, kept by ``install_lerobot_patches``."""


def install_lerobot_patches() -> None:
    """Install the uint8 decoders and the deduplicated video check. Idempotent."""
    global _installed, original_video_check
    if _installed:
        return
    import lerobot.datasets.video_utils as vu
    from lerobot.datasets.dataset_reader import DatasetReader

    warnings.filterwarnings(
        "ignore",
        message="The video decoding and encoding capabilities of torchvision are deprecated",
        category=UserWarning,
    )
    vu.decode_video_frames_torchvision = _decode_video_frames_torchvision_uint8
    vu.decode_video_frames_torchcodec = _decode_video_frames_torchcodec_uint8
    original_video_check = DatasetReader._check_cached_episodes_sufficient
    DatasetReader._check_cached_episodes_sufficient = dedup_video_check  # type: ignore[method-assign]
    _installed = True
    logger.info("Installed the LeRobot patches (uint8 decoders, deduplicated video check).")


@contextlib.contextmanager
def thread_safe_tqdm_lock() -> Iterator[None]:
    """Make ``tqdm.contrib.concurrent.thread_map`` (huggingface_hub's snapshot download, inside
    ``LeRobotDataset``) safe to call from several threads at once.

    Its ``ensure_lock`` deletes ``tqdm._lock`` on exit when it created it, and a concurrent
    ``tqdm.__init__`` that finds no class-level lock crashes. The replacement never deletes it.
    """
    import tqdm.contrib.concurrent as tcc
    import tqdm.std

    if not hasattr(tqdm.std.tqdm, "_lock"):
        tqdm.std.tqdm.get_lock()

    @contextlib.contextmanager
    def ensure_lock(tqdm_class: type, lock_name: str = "") -> Iterator[object]:
        old = getattr(tqdm_class, "_lock", None)
        lock = old or tqdm_class.get_lock()  # type: ignore[attr-defined]
        lock = getattr(lock, lock_name, lock)
        tqdm_class.set_lock(lock)  # type: ignore[attr-defined]
        try:
            yield lock
        finally:
            if old is not None:
                tqdm_class.set_lock(old)  # type: ignore[attr-defined]

    original = tcc.ensure_lock  # type: ignore[attr-defined]
    tcc.ensure_lock = ensure_lock  # type: ignore[assignment]
    try:
        yield
    finally:
        tcc.ensure_lock = original  # type: ignore[attr-defined]


class _BoundedDecoderCache:
    """LRU of torchcodec decoders keyed by video path, closing evicted file handles.

    LeRobot's default ``VideoDecoderCache`` is an unbounded dict, so a worker touching
    thousands of leaf videos leaks decoders and open handles until it dies.
    """

    def __init__(self, maxsize: int = 8) -> None:
        from collections import OrderedDict
        from threading import Lock

        self._maxsize = maxsize
        self._cache: OrderedDict[str, tuple[Any, Any]] = OrderedDict()
        self._lock = Lock()

    def get_decoder(self, video_path: str):
        import fsspec
        from torchcodec.decoders import VideoDecoder

        with self._lock:
            if video_path in self._cache:
                self._cache.move_to_end(video_path)
                return self._cache[video_path][0]
            file_handle = fsspec.open(video_path).__enter__()
            decoder = VideoDecoder(file_handle, seek_mode="approximate")
            self._cache[video_path] = (decoder, file_handle)
            while len(self._cache) > self._maxsize:
                _, (_, old_handle) = self._cache.popitem(last=False)
                old_handle.close()
            return decoder


_bounded_decoder_cache = _BoundedDecoderCache()


def _decode_video_frames_torchvision_uint8(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str = "pyav",
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """uint8 variant of lerobot.datasets.video_utils.decode_video_frames_torchvision."""
    import torchvision
    from lerobot.datasets.video_utils import FrameTimestampError

    video_path = str(video_path)
    keyframes_only = False
    torchvision.set_video_backend(backend)
    if backend == "pyav":
        keyframes_only = True

    reader = torchvision.io.VideoReader(video_path, "video")
    first_ts = min(timestamps)
    last_ts = max(timestamps)
    reader.seek(first_ts, keyframes_only=keyframes_only)

    loaded_frames: list[torch.Tensor] = []
    loaded_ts: list[float] = []
    for frame in reader:
        current_ts = frame["pts"]
        if log_loaded_timestamps:
            logger.info(f"frame loaded at timestamp={current_ts:.4f}")
        loaded_frames.append(frame["data"])
        loaded_ts.append(current_ts)
        if current_ts >= last_ts:
            break

    if backend == "pyav":
        reader.container.close()
    reader = None

    if not loaded_ts:
        raise FrameTimestampError(
            f"Video decoder returned 0 frames after seeking to {first_ts:.3f}s "
            f"(queried {timestamps}, backend={backend}, video={video_path})."
        )

    query_ts = torch.tensor(timestamps)
    loaded_ts_t = torch.tensor(loaded_ts)
    dist = torch.cdist(query_ts[:, None], loaded_ts_t[:, None], p=1)
    min_, argmin_ = dist.min(1)
    is_within_tol = min_ < tolerance_s
    if not is_within_tol.all():
        raise FrameTimestampError(
            f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
            f"\nqueried timestamps: {query_ts}"
            f"\nloaded timestamps: {loaded_ts_t}"
            f"\nvideo: {video_path}"
            f"\nbackend: {backend}"
        )

    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    if len(timestamps) != len(closest_frames):
        raise FrameTimestampError(
            f"Number of retrieved frames ({len(closest_frames)}) does not match "
            f"number of queried timestamps ({len(timestamps)})"
        )
    return closest_frames


def _decode_video_frames_torchcodec_uint8(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    log_loaded_timestamps: bool = False,
    decoder_cache: Any | None = None,
) -> torch.Tensor:
    """uint8 variant of lerobot.datasets.video_utils.decode_video_frames_torchcodec."""
    from lerobot.datasets.video_utils import FrameTimestampError

    if decoder_cache is None:
        decoder_cache = _bounded_decoder_cache

    decoder = decoder_cache.get_decoder(str(video_path))
    loaded_ts: list[float] = []
    loaded_frames: list[torch.Tensor] = []

    metadata = decoder.metadata
    average_fps = metadata.average_fps
    frame_indices = [round(ts * average_fps) for ts in timestamps]
    frames_batch = decoder.get_frames_at(indices=frame_indices)

    for frame, pts in zip(frames_batch.data, frames_batch.pts_seconds, strict=True):
        loaded_frames.append(frame)
        loaded_ts.append(pts.item())
        if log_loaded_timestamps:
            logger.info(f"Frame loaded at timestamp={pts:.4f}")

    if not loaded_ts:
        raise FrameTimestampError(
            f"Video decoder returned 0 frames (queried {timestamps}, video={video_path})."
        )

    query_ts = torch.tensor(timestamps)
    loaded_ts_t = torch.tensor(loaded_ts)
    dist = torch.cdist(query_ts[:, None], loaded_ts_t[:, None], p=1)
    min_, argmin_ = dist.min(1)
    is_within_tol = min_ < tolerance_s
    if not is_within_tol.all():
        raise FrameTimestampError(
            f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
            f"\nqueried timestamps: {query_ts}"
            f"\nloaded timestamps: {loaded_ts_t}"
            f"\nvideo: {video_path}"
        )

    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    if not len(timestamps) == len(closest_frames):
        raise FrameTimestampError(
            f"Retrieved timestamps differ from queried ({len(closest_frames)} vs {len(timestamps)})"
        )
    return closest_frames


def dedup_video_check(self: Any) -> bool:
    """LeRobot's ``_check_cached_episodes_sufficient``, checking each shared video file once:
    whether the requested episodes and their video files are cached."""
    import torch
    from lerobot.datasets.io_utils import load_episodes

    if self.hf_dataset is None or len(self.hf_dataset) == 0:
        return False

    available_episodes = {
        ep_idx.item() if isinstance(ep_idx, torch.Tensor) else ep_idx
        for ep_idx in self.hf_dataset.unique("episode_index")
    }
    if self.episodes is None:
        requested_episodes = set(range(self._meta.total_episodes))
    else:
        requested_episodes = set(self.episodes)
    if not requested_episodes.issubset(available_episodes):
        return False

    meta = self._meta
    if len(meta.video_keys) > 0 and requested_episodes:
        if meta.episodes is None:
            meta.episodes = load_episodes(meta.root)
        n_eps = len(meta.episodes)
        chunks = {k: list(meta.episodes[f"videos/{k}/chunk_index"]) for k in meta.video_keys}
        files = {k: list(meta.episodes[f"videos/{k}/file_index"]) for k in meta.video_keys}
        checked: set[str] = set()
        for ep_idx in requested_episodes:
            if ep_idx >= n_eps:
                raise IndexError(f"Episode index {ep_idx} out of range. Episodes: {n_eps}")
            for vid_key in meta.video_keys:
                fpath = meta.video_path.format(
                    video_key=vid_key,
                    chunk_index=chunks[vid_key][ep_idx],
                    file_index=files[vid_key][ep_idx],
                )
                if fpath in checked:
                    continue
                if not (self.root / fpath).exists():
                    return False
                checked.add(fpath)

    return True
