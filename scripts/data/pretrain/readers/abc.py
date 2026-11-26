"""Reader for ABC-130k (XDOF bimanual YAM teleop, one MCAP per episode) → two collections,
routed per episode by camera station, one leaf per task:

- ``abc_yam_realsense``: ``/top-camera`` + 2 wrists, 640×480.
- ``abc_yam_zedx``: ``/top-left-camera`` (right eye unused) + 2 wrists, 1920×1200 → 768×480.

State = action = 14-D ``[Larm6, Lgrip, Rarm6, Rgrip]`` (``/{l,r}-arm-*`` + ``/{l,r}-ee-*``).
Task text = the task folder name, capitalised, with a final period.

Resample, per episode (all times = MCAP ``log_time``): ticks ``t0 + k/30 s`` over the span common
to every stream used (3 cameras, 8 state/action topics). Each camera takes its newest frame at
or before the tick; each state/action topic (gripper included) is linearly interpolated at the
tick. ``annotation.mcap`` subtask starts map to the first tick at or after them; the literal
``"mistake"`` marks the segment before it as a failed attempt (``mistake_intervals``).

Decode: NVDEC only (ffmpeg ``*_cuvid`` + ``scale_cuda``), so a build needs one GPU; decoded
frames must equal the packet count. The decoded nv12 frames go straight into CPU libx264
(trained settings, no RGB round trip): the pixels are ~0.5 level from the source. Any read
error raises; the only skipped episodes are ``ABC_SKIP``.

Build is two steps: ``convert`` turns each episode into arrays + per-camera mp4s under
``<root>/.work/abc/`` (sharded by episode, resumable), then ``assemble`` writes the leaves
(tasks in parallel). One convert process uses ~4 CPU cores and a small share of NVDEC, so run
about one ``build --shard i/n`` process per 2 cores, many per GPU, each with
``CUDA_VISIBLE_DEVICES`` set to its GPU; the last shard to end assembles.
``curate`` writes each leaf's ``meta/subtask_lookup.jsonl`` from the same work dirs.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import shutil
import subprocess
import zlib
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import BinaryIO

import numpy as np

from ..common import v30
from ..common.download import upstream_dir
from ..common.stats import op
from ..common.video import video_frames
from ..pins import HfSnapshot, Pin
from ..recipe import Reader, Recipe, collection_dir

logger = logging.getLogger(__name__)

FPS = 30
SECOND_NS = 1_000_000_000
X264 = ("-c:v", "libx264", "-preset", "ultrafast", "-g", "2", "-crf", "30", "-pix_fmt", "yuv420p")
"""The encoder settings of the trained leaves."""

STATIONS = {
    "abc_yam_realsense": ("/top-camera", (480, 640)),
    "abc_yam_zedx": ("/top-left-camera", (480, 768)),
}
WRISTS = {"camera_wrist_left": "/left-wrist-camera", "camera_wrist_right": "/right-wrist-camera"}
CAMERAS = ("camera_top", *WRISTS)
STATE_TOPICS = ("/left-arm-state", "/left-ee-state", "/right-arm-state", "/right-ee-state")
ACTION_TOPICS = ("/left-arm-action", "/left-ee-action", "/right-arm-action", "/right-ee-action")
ANNOTATION_TOPIC = "/subtask-annotation"
MISTAKE = "mistake"
EPISODES_PER_FFMPEG = 4
"""Episodes converted by one ffmpeg (fewer CUDA set-ups and tear-downs per node)."""
ASSEMBLE_WORKERS = 16
"""Leaves written at once by ``assemble`` (each task's leaf is independent; I/O-bound)."""


# ------------------------------------------------------------------ resample rule


def tick_grid(first: Sequence[int], last: Sequence[int]) -> np.ndarray:
    """The 30 Hz ticks (int64 ns) over the span common to all streams, given each stream's
    first and last timestamp: ``t0 + floor(k * 1e9 / 30)`` for every tick ``<= t_end``."""
    t0, t_end = max(first), min(last)
    if t_end < t0:
        raise ValueError(f"streams do not overlap: common span [{t0}, {t_end}]")
    k = np.arange(((t_end - t0 + 1) * FPS - 1) // SECOND_NS + 1, dtype=np.int64)
    return t0 + k * SECOND_NS // FPS


def floor_indices(ts: np.ndarray, ticks: np.ndarray) -> np.ndarray:
    """Per tick, the index of the newest ``ts`` at or before it."""
    idx = np.searchsorted(ts, ticks, side="right") - 1
    if (idx < 0).any():
        raise ValueError("a tick precedes the first frame")
    return idx


def interpolate(ts: np.ndarray, values: np.ndarray, ticks: np.ndarray) -> np.ndarray:
    """``values`` ``[N, D]`` stamped ``ts`` linearly interpolated at ``ticks`` (float32)."""
    if ticks[0] < ts[0] or ticks[-1] > ts[-1]:
        raise ValueError("ticks outside the topic's span")
    x, xp = (ticks - ticks[0]).astype(np.float64), (ts - ticks[0]).astype(np.float64)
    v = values.astype(np.float64)
    return np.stack([np.interp(x, xp, v[:, d]) for d in range(v.shape[1])], 1).astype(np.float32)


def subtask_intervals(
    annotations: Sequence[tuple[int, str]], ticks: np.ndarray
) -> tuple[list[list], list[int]]:
    """``(log_time, text)`` subtask starts → ``[[start, end, text]]`` tick intervals and the
    indices of the intervals followed by a ``"mistake"`` marker.

    A start maps to the first tick at or after it; a segment ends where the next annotation
    starts (the last one at the episode end). The marker itself is no interval. Empty
    segments are dropped."""
    anns = sorted(annotations, key=lambda a: a[0])
    starts = np.searchsorted(ticks, np.array([t for t, _ in anns], dtype=np.int64), side="left")
    n = len(ticks)
    intervals: list[list] = []
    mistakes: list[int] = []
    emitted: dict[int, int] = {}
    for i, (_, text) in enumerate(anns):
        if text.lower() == MISTAKE:
            if i - 1 in emitted:
                mistakes.append(emitted[i - 1])
            continue
        start, end = int(starts[i]), int(starts[i + 1]) if i + 1 < len(anns) else n
        if end > start:
            emitted[i] = len(intervals)
            intervals.append([start, end, text])
    return intervals, sorted(set(mistakes))


def task_text(task: str) -> str:
    """``wrap_the_cables`` → ``Wrap the cables.``"""
    s = task.replace("_", " ").strip()
    return s[0].upper() + s[1:] + ("" if s.endswith((".", "!", "?")) else ".")


# ------------------------------------------------------------------ MCAP


def _reader(f: BinaryIO):  # noqa: ANN202
    from mcap.reader import make_reader  # type: ignore[import-not-found]
    from mcap_protobuf.decoder import DecoderFactory  # type: ignore[import-not-found]

    return make_reader(f, decoder_factories=[DecoderFactory()])


def _open(mcap: Path | bytes) -> BinaryIO:
    return open(mcap, "rb") if isinstance(mcap, Path) else io.BytesIO(mcap)


def station_of(mcap: Path | bytes) -> str:
    """The registry dataset of an episode, from the channels in its MCAP summary (the file, or
    its bytes)."""
    with _open(mcap) as f:
        summary = _reader(f).get_summary()
    what = mcap if isinstance(mcap, Path) else "MCAP"
    if summary is None:
        raise ValueError(f"{what}: no MCAP summary")
    topics = {c.topic for c in summary.channels.values()}
    names = [name for name, (top, _) in STATIONS.items() if top in topics]
    if len(names) != 1:
        raise ValueError(f"{what}: station unknown, topics {sorted(topics)}")
    return names[0]


def _stamps(rows: list[tuple[int, object]], what: str) -> np.ndarray:
    ts = np.array([t for t, _ in rows], dtype=np.int64)
    if len(ts) < 2 or (np.diff(ts) <= 0).any():
        raise ValueError(f"{what}: {len(ts)} messages, log_time not strictly increasing")
    return ts


def read_episode(mcap: Path | bytes, name: str) -> tuple[dict, dict, dict]:
    """``(scalars, cameras, codecs)``: per topic ``(log_time, value)`` rows in file order (from
    the file, or its bytes)."""
    top, _ = STATIONS[name]
    cams = {"camera_top": top, **WRISTS}
    scalar_topics = {*STATE_TOPICS, *ACTION_TOPICS}
    scalars: dict[str, list] = {t: [] for t in scalar_topics}
    cameras: dict[str, list] = {k: [] for k in cams}
    by_topic = {t: k for k, t in cams.items()}
    codecs: dict[str, str] = {}
    with _open(mcap) as f:
        for _, channel, message, decoded in _reader(f).iter_decoded_messages(
            topics=[*scalar_topics, *cams.values()]
        ):
            if channel.topic in scalar_topics:
                scalars[channel.topic].append(
                    (message.log_time, np.asarray(decoded.position, np.float64))
                )
            else:
                key = by_topic[channel.topic]
                cameras[key].append((message.log_time, bytes(decoded.data)))
                codecs[key] = "hevc" if decoded.format.lower() in ("h265", "hevc") else "h264"
    return scalars, cameras, codecs


def read_annotations(path: Path) -> list[tuple[int, str]]:
    """The ``/subtask-annotation`` rows of an ``annotation.mcap`` (none without the file)."""
    if not path.exists():
        return []
    with open(path, "rb") as f:
        return [
            (message.log_time, str(decoded.data).strip())
            for _, _, message, decoded in _reader(f).iter_decoded_messages(
                topics=[ANNOTATION_TOPIC]
            )
        ]


# ------------------------------------------------------------------ video


SUB = 1 << 20
"""Sub-steps per output frame in the time base of the timed streams (``1 / (FPS * SUB)`` s)."""


def slot_pts(picks: np.ndarray) -> np.ndarray:
    """Per native frame up to the last pick, a pts in ``1 / (FPS * SUB)`` s such that, for every
    output frame ``k``, the newest native frame with ``round(pts / SUB) <= k`` is ``picks[k]``
    (what ffmpeg's ``fps`` filter keeps). A shown frame sits at ``SUB *`` its first output frame;
    the frames no output frame shows sit just before the next shown one. Strictly increasing."""
    n = np.arange(picks[-1] + 1)
    first = np.searchsorted(picks, n, side="left")  # the first output frame showing >= n
    # a run of equal ``first`` is hidden frames, then the shown one: step the hidden ones back
    ends = np.flatnonzero(np.r_[first[1:] != first[:-1], True])
    back = ends[np.searchsorted(ends, n)] - n
    if back.max() >= SUB // 2:
        raise ValueError(f"{back.max()} hidden frames before one output frame")
    return first * SUB - back


def _template(packets: list[bytes], codec: str):  # noqa: ANN202
    """A demuxer of the first packets of an Annex-B stream (its stream holds the codec params)."""
    import av  # type: ignore[import-not-found]

    return av.open(io.BytesIO(b"".join(packets[:8])), format=codec)


def _write_timed(packets: list[bytes], codec: str, picks: np.ndarray, fd: int) -> None:
    """Stream-copy the Annex-B ``packets`` up to the last pick as NUT into ``fd`` (closed
    after), with :func:`slot_pts` timestamps (one packet = one access unit = one frame; no
    B-frames, so the later frames are never references)."""
    from fractions import Fraction

    import av  # type: ignore[import-not-found]

    pts = slot_pts(picks)
    end = len(picks) * SUB  # the last frame lasts to the end of the grid
    with open(fd, "wb") as f, _template(packets, codec) as probe:
        with av.open(f, "w", format="nut") as out:
            stream = out.add_stream_from_template(probe.streams.video[0], opaque=True)
            stream.time_base = Fraction(1, FPS * SUB)
            for i, data in enumerate(packets[: len(pts)]):
                packet = av.Packet(data)
                packet.stream = stream
                packet.time_base = stream.time_base
                packet.pts = packet.dts = int(pts[i])
                packet.duration = int(pts[i + 1] if i + 1 < len(pts) else end) - int(pts[i])
                out.mux(packet)


_RAW_PARAMS = (
    "setparams=range=unknown:color_primaries=unknown:color_trc=unknown:colorspace=unknown,setsar=0"
)
"""The frame metadata of a raw nv12 input (the encoder input of the per-frame path)."""


NVDEC = ("-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-hwaccel_device", "cu")
"""An NVDEC input on the process's CUDA device ``cu``."""


def _cpu_decodable(codec: str, profile: str, coded: tuple[int, int], hw: tuple[int, int]) -> bool:
    """Baseline H.264 already at the output size: the CPU decode gives the NVDEC frames (H.264
    decode is bit-exact) and ``scale_cuda`` would be a no-op. Baseline has no B-frames, which the
    CPU path does not reorder (the ABC realsense streams are Constrained Baseline)."""
    return codec == "h264" and "Baseline" in profile and coded == hw


Camera = tuple[list[bytes], str, np.ndarray, tuple[int, int], Path]
"""``(packets, codec, picks, hw, out)``: an Annex-B stream, the native frame of each output
frame, the output size and mp4."""


def resample_videos(cams: list[Camera]) -> None:
    """Encode each camera: decode its Annex-B ``packets`` (``codec``), write native frame
    ``picks[i]`` at output frame ``i`` (libx264, 30 fps, size ``hw``). Frames stay in ffmpeg:
    each stream is piped in with :func:`slot_pts` timestamps and ffmpeg's ``fps`` filter picks
    and repeats them. H.264 already at ``hw`` decodes on the CPU, in one ffmpeg; the rest in
    another, with NVDEC + ``scale_cuda`` on one CUDA device (the NVIDIA driver serializes CUDA
    work across a node, so few CUDA processes beat many). Same frames, settings and bytes as an
    NVDEC decode → nv12 → libx264 of each picked frame. Every packet sent (up to the last pick)
    must decode to one frame."""
    cpu, gpu = [], []
    for packets, codec, picks, hw, out in cams:
        with _template(packets, codec) as probe:
            ctx = probe.streams.video[0].codec_context
            coded, profile = (ctx.height, ctx.width), ctx.profile or ""
        (cpu if _cpu_decodable(codec, profile, coded, hw) else gpu).append(
            (packets, codec, picks, hw, out)
        )
    with ThreadPoolExecutor(2) as pool:
        runs = [pool.submit(_ffmpeg, group, group is gpu) for group in (cpu, gpu) if group]
        for run in runs:
            run.result()


def _ffmpeg(cams: list[Camera], nvdec: bool) -> None:
    """One ffmpeg for :func:`resample_videos`: all ``cams`` decoded on the CPU, or with NVDEC."""
    inputs, graphs, outs, pipes = [], [], [], []
    for k, (packets, codec, picks, (h, w), out) in enumerate(cams):
        assert h % 2 == 0 and w % 2 == 0, f"nv12 needs an even size, got {w}x{h}"
        read, write = os.pipe()
        pipes.append((read, write))
        if nvdec:
            inputs += [*NVDEC, "-c:v", f"{codec}_cuvid", "-i", f"pipe:{read}"]
            chain = f"scale_cuda={w}:{h},hwdownload,format=nv12,"
        else:
            inputs += ["-threads", "1", "-c:v", "h264", "-i", f"pipe:{read}"]
            chain = ""
        # fps first: hidden frames are dropped before scaling. The input ends with the last
        # pick, whose duration NUT does not keep: fps passes it once at the end and tpad
        # repeats it until -frames:v
        graphs.append(
            f"[{k}:v]fps={FPS}:eof_action=pass,{chain}{_RAW_PARAMS},"
            f"tpad=stop=-1:stop_mode=clone[v{k}]"
        )
        outs += ["-map", f"[v{k}]", "-frames:v", str(len(picks)), *X264, "-threads", "1"]
        outs += ["-map_metadata", "-1", "-y", str(out)]
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "verbose", "-filter_threads", "1"]
    if nvdec:
        cmd += ["-init_hw_device", "cuda=cu:0"]
    cmd += [*inputs, "-filter_complex", ";".join(graphs), *outs]
    proc = subprocess.Popen(cmd, pass_fds=[r for r, _ in pipes], stderr=subprocess.PIPE, text=True)
    for read, _ in pipes:
        os.close(read)
    with ThreadPoolExecutor(len(cams)) as pool:  # ffmpeg reads the inputs concurrently
        writes = [
            pool.submit(_write_timed, packets, codec, picks, write)
            for (packets, codec, picks, _, _), (_, write) in zip(cams, pipes, strict=True)
        ]
        _, err = proc.communicate()
    failed = next((e for e in (f.exception() for f in writes) if e is not None), None)
    if proc.returncode != 0 or failed is not None:
        raise RuntimeError(f"{cams[0][4].parent}: ffmpeg failed: {err[-600:]}") from failed
    decoded = [int(m) for m in re.findall(r"(\d+) frames decoded; 0 decode errors", err)]
    sent = [int(picks[-1]) + 1 for _, _, picks, _, _ in cams]
    if decoded != sent:
        raise RuntimeError(
            f"{cams[0][4].parent}: decoded {decoded} frames from {sent} packets: {err[-600:]}"
        )


# ------------------------------------------------------------------ episodes


def _episode_arrays(ep_dir: Path, name: str, mcap: Path | bytes) -> tuple[dict, dict, dict]:
    """``(arrays, cameras, codecs)`` of one episode: the npz arrays and, per camera, its packets
    and the native frame index of each tick."""
    scalars, cameras, codecs = read_episode(mcap, name)
    stamps = {t: _stamps(rows, f"{ep_dir.name} {t}") for t, rows in scalars.items()}
    cam_ts = {k: _stamps(rows, f"{ep_dir.name} {k}") for k, rows in cameras.items()}
    streams = [*stamps.values(), *cam_ts.values()]
    ticks = tick_grid([s[0] for s in streams], [s[-1] for s in streams])

    def vec(topics: Sequence[str]) -> np.ndarray:
        parts = []
        for t in topics:
            values = np.stack([np.atleast_1d(v) for _, v in scalars[t]])
            parts.append(interpolate(stamps[t], values if "-arm-" in t else values[:, :1], ticks))
        return np.concatenate(parts, axis=1)

    state, action = vec(STATE_TOPICS), vec(ACTION_TOPICS)
    if state.shape[1] != 14 or action.shape[1] != 14:
        raise ValueError(f"{ep_dir.name}: state {state.shape}, action {action.shape}")
    if not (np.isfinite(state).all() and np.isfinite(action).all()):
        raise ValueError(f"{ep_dir.name}: non-finite state/action")
    intervals, mistakes = subtask_intervals(read_annotations(ep_dir / "annotation.mcap"), ticks)
    arrays = {
        "state": state,
        "action": action,
        "ticks": ticks,
        "intervals": np.array(json.dumps({"intervals": intervals, "mistakes": mistakes})),
    }
    cams = {
        k: ([d for _, d in rows], floor_indices(cam_ts[k], ticks)) for k, rows in cameras.items()
    }
    return arrays, cams, codecs


def convert_episodes(jobs: Sequence[tuple[Path, str, Path, bytes | None]]) -> list[int]:
    """Episodes ``(ep_dir, station, out, mcap bytes or None: read ep_dir/episode.mcap)`` →
    ``out/{arrays.npz, <camera>.mp4}``, all cameras in one ffmpeg; returns the grid lengths."""
    episodes, cams = [], []
    for ep_dir, name, out, mcap in jobs:
        arrays, cameras, codecs = _episode_arrays(ep_dir, name, mcap or ep_dir / "episode.mcap")
        out.mkdir(parents=True, exist_ok=True)
        _, hw = STATIONS[name]
        cams += [(p, codecs[k], picks, hw, out / f"{k}.mp4") for k, (p, picks) in cameras.items()]
        episodes.append((out, arrays, list(cameras)))
    resample_videos(cams)
    for out, arrays, keys in episodes:
        for k in keys:
            if video_frames(out / f"{k}.mp4") != len(arrays["ticks"]):
                raise RuntimeError(
                    f"{out / k}.mp4: {video_frames(out / f'{k}.mp4')} frames, grid {len(arrays['ticks'])}"
                )
        np.savez(out / "arrays.npz", **arrays)
    return [len(arrays["ticks"]) for _, arrays, _ in episodes]


def _assemble_task(name: str, root: Path, task: str, eps: list[Path]) -> None:
    """The leaf of one task from its converted episodes."""

    def episodes() -> Iterator[v30.Episode]:
        for new, ep in enumerate(eps):
            with np.load(ep / "arrays.npz") as z:
                state, action = z["state"], z["action"]
            videos = {f"observation.images.{k}": ep / f"{k}.mp4" for k in CAMERAS}
            yield v30.Episode(new, state, action, task_text(task), video_src=videos)

    v30.write_leaf(episodes(), collection_dir(name, root) / task, FPS, "yam")
    logger.info("%s %s: %d episodes", name, task, len(eps))


def pinned_episodes(pin: Pin) -> dict[str, list[str] | None]:
    """Task → the pinned episode dirs (sorted), from the pin's allow-patterns
    ``data/train/<task>/*`` (None: every episode of the task) or ``data/train/<task>/<ep>/*``."""
    (source,) = pin.sources
    assert isinstance(source, HfSnapshot) and source.allow is not None
    out: dict[str, list[str] | None] = {}
    for pattern in source.allow:
        parts = pattern.split("/")
        if len(parts) == 4:
            out[parts[2]] = None
        else:
            eps = out.setdefault(parts[2], [])
            assert eps is not None
            eps.append(parts[3])
    return {t: None if eps is None else sorted(eps) for t, eps in sorted(out.items())}


class Abc(Reader):
    """One station of ABC-130k → one ``<collection>/<task>`` leaf per pinned task."""

    def leaves(self, name: str, root: Path, pin: Pin) -> list[Path]:
        return [collection_dir(name, root) / t for t in pinned_episodes(pin)]

    def _work(self, root: Path, name: str, task: str) -> Path:
        return root / ".work" / "abc" / name / task

    def episodes(self, root: Path, pin: Pin, task: str) -> list[Path]:
        """The episode dirs of a task: the pinned ones, or every dir minus ``ABC_SKIP``, whose
        count must match ``ABC_TASKS``."""
        (source,) = pin.sources
        task_dir = upstream_dir(root, source) / "data" / "train" / task
        names = pinned_episodes(pin)[task]
        if names is None:
            names = sorted(p.name for p in task_dir.iterdir() if p.name.startswith("episode_"))
            names = [n for n in names if n.removeprefix("episode_") not in ABC_SKIP]
            if len(names) != sum(ABC_TASKS[task]):
                raise ValueError(
                    f"{task_dir}: {len(names)} episodes, pinned {sum(ABC_TASKS[task])}"
                )
        return [task_dir / n for n in names]

    def shard_episodes(
        self, root: Path, pin: Pin, shard: tuple[int, int]
    ) -> list[tuple[str, Path]]:
        """Shard ``(i, n)``: the (task, episode dir) of all pinned tasks whose name hashes to i
        mod n, so the shards of one ``n`` are disjoint, an episode is in the same shard for both
        stations' pins, and no task bounds the wall time."""
        i, n = shard
        return [
            (t, ep)
            for t in pinned_episodes(pin)
            for ep in self.episodes(root, pin, t)
            if zlib.crc32(f"{t}/{ep.name}".encode()) % n == i
        ]

    def convert(self, name: str, root: Path, pin: Pin, shard: tuple[int, int]) -> None:
        """Step 1 (sharded by episode, resumable): each episode of the shard → the work dir of its
        station, whichever ``name`` is (the station comes from the summary of the MCAP bytes read
        for the conversion, so no read is spent on routing). All shards of a run use the same
        ``n``; every task must be downloaded."""
        todo = [
            (task, ep_dir)
            for task, ep_dir in self.shard_episodes(root, pin, shard)
            if not self._station_done(root, task, ep_dir.name)
        ]
        for start in range(0, len(todo), EPISODES_PER_FFMPEG):
            jobs, works = [], []
            for task, ep_dir in todo[start : start + EPISODES_PER_FFMPEG]:
                mcap = (ep_dir / "episode.mcap").read_bytes()
                station = station_of(mcap)
                work = self._work(root, station, task) / ep_dir.name
                tmp = work.with_name(work.name + ".part")
                if tmp.exists():
                    shutil.rmtree(tmp)
                jobs.append((ep_dir, station, tmp, mcap))
                works.append((station, task, work))
            lengths = convert_episodes(jobs)
            for (_, _, tmp, _), (station, task, work), length in zip(jobs, works, lengths):
                tmp.rename(work)
                logger.info("%s %s %s: %d ticks", station, task, work.name, length)

    def _station_done(self, root: Path, task: str, episode: str) -> str | None:
        """The station whose work dir holds a converted ``episode``, if any."""
        done = [s for s in STATIONS if (self._work(root, s, task) / episode).exists()]
        if len(done) > 1:
            raise ValueError(f"{task} {episode}: converted for {done}")
        return done[0] if done else None

    def converted(self, name: str, root: Path, pin: Pin) -> dict[str, list[Path]]:
        """Task → the work dirs of this station's episodes, in leaf order; all must exist."""
        out: dict[str, list[Path]] = {}
        for task in pinned_episodes(pin):
            done = {
                ep.name: self._station_done(root, task, ep.name)
                for ep in self.episodes(root, pin, task)
            }
            missing = [ep for ep, station in done.items() if station is None]
            eps = [
                self._work(root, name, task) / ep for ep, station in done.items() if station == name
            ]
            if missing:
                raise FileNotFoundError(f"{name} {task}: {len(missing)} episodes not converted")
            if eps:
                out[task] = eps
        if sum(map(len, out.values())) != pin.episodes:
            raise ValueError(
                f"{name}: {sum(map(len, out.values()))} episodes, pinned {pin.episodes}"
            )
        return out

    def assemble(self, name: str, root: Path, pin: Pin) -> None:
        """Step 2: one leaf per task, ``ASSEMBLE_WORKERS`` tasks at a time."""
        converted = self.converted(name, root, pin)
        with ProcessPoolExecutor(min(ASSEMBLE_WORKERS, len(converted))) as pool:
            list(pool.map(partial(_assemble_task, name, root), converted, converted.values()))

    def curate(self, name: str, root: Path, pin: Pin) -> None:
        """Each leaf's ``meta/subtask_lookup.jsonl`` from the converted annotation intervals."""
        for task, eps in self.converted(name, root, pin).items():
            rows = []
            for index, ep in enumerate(eps):
                with np.load(ep / "arrays.npz") as z:
                    labels = json.loads(str(z["intervals"]))
                if labels["intervals"]:
                    row: dict = {"episode_index": index, "intervals": labels["intervals"]}
                    if labels["mistakes"]:
                        row["mistake_intervals"] = labels["mistakes"]
                    rows.append(row)
            path = collection_dir(name, root) / task / "meta" / "subtask_lookup.jsonl"
            path.unlink(missing_ok=True)
            if rows:
                path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            logger.info("%s %s: %d episodes with subtasks", name, task, len(rows))

    finish_after_shards = True

    def build(self, name: str, root: Path, pin: Pin, shard: tuple[int, int]) -> None:
        """A shard converts its episodes; the unsharded run converts what is left, then assembles."""
        self.convert(name, root, pin, shard)
        if shard == (0, 1):
            self.assemble(name, root, pin)


# ------------------------------------------------------------------ ABC-130k

ABC_REVISION = "d4c79f5aecb902f3157e6139661cb1b0b2a36689"
"""XDOF/ABC-130k: the 197 trained tasks (the 4 tasks added upstream are not used)."""

ABC_SKIP = (
    "72d3ae3b-2245-4aa2-b998-b85c431b552b",
    "da186ce1-037d-4ea4-a8ad-88009f3a0d33",
    "c29c86b1-dbf1-42d9-8b6c-61532459fab4",
    "c61e1cfd-34b1-4276-987a-792fb9ba554e",
)
"""Episode uuids left out (put_the_pillow_into_the_pillowcase ×2, pack_the_luggage ×2), as in
the trained leaves."""

ABC_TASKS: dict[str, tuple[int, int]] = {
    "arrange_the_flowers_into_the_vase": (0, 203),
    "assemble_a_carrot_with_lego": (36, 0),
    "attach_the_microphones_to_the_stand": (235, 0),
    "build_the_wood_block_tower": (360, 239),
    "clean_the_litter_box": (172, 0),
    "clear_the_kitchen_counter": (90, 0),
    "clip_the_socks_to_the_hanger": (457, 0),
    "clip_the_underwear_to_the_hanger": (151, 0),
    "connect_and_route_the_hose": (75, 0),
    "decorate_the_small_christmas_tree": (180, 0),
    "distribute_texas_hold_em_gaming_equipment": (421, 0),
    "dress_the_teddy_bear": (0, 70),
    "erase_the_whiteboard": (1276, 0),
    "fill_the_litter_box_with_clean_litter": (216, 0),
    "fold_a_paper_plane_with_a_square_sheet_of_paper_fold_left_side_to_right_side": (570, 21),
    "fold_a_paper_plane_with_a_square_sheet_of_paper_pick_it_up_from_the_middle": (1735, 30),
    "fold_and_stack_the_long_sleeve_shirts": (546, 85),
    "fold_and_stack_the_mixed_laundry_pile": (254, 24),
    "fold_and_stack_the_shorts": (637, 116),
    "fold_and_stack_the_skirts": (1124, 57),
    "fold_and_stack_the_t_shirts": (10791, 218),
    "fold_and_stack_the_tank_tops": (611, 92),
    "fold_and_stack_the_towels": (290, 94),
    "fold_and_stack_the_trousers": (529, 78),
    "fold_the_inside_out_t_shirt": (527, 0),
    "fold_the_napkin_into_a_case_and_place_the_utensils_inside": (470, 0),
    "fold_the_napkin_place_the_utensils_inside_and_roll_it_up": (399, 0),
    "fold_the_paper_box": (2234, 153),
    "insert_the_pens_into_the_pen_caps": (527, 257),
    "insert_the_plug": (627, 181),
    "insert_the_plug_into_the_switch_port": (317, 110),
    "insert_the_wireless_bluetooth_earbuds_into_the_charging_case": (1973, 122),
    "install_the_large_spring": (114, 0),
    "install_the_water_faucets": (352, 0),
    "load_the_batteries_into_the_remote_control": (1040, 127),
    "load_the_bowls_into_the_dish_rack": (458, 143),
    "load_the_cups_into_the_dish_rack": (384, 93),
    "load_the_mixed_dishes_into_the_dish_rack": (568, 0),
    "load_the_plates_into_the_dish_rack": (1240, 95),
    "load_the_staples_into_the_stapler": (400, 147),
    "lock_with_the_key": (440, 196),
    "open_and_lay_out_the_surgical_kit": (307, 0),
    "open_take_out_and_arrange_the_luggage": (182, 0),
    "open_the_pen_caps": (459, 57),
    "open_the_umbrella": (106, 0),
    "open_the_zip_top_bag_and_remove_the_fake_food": (763, 0),
    "organize_the_chemistry_lab_equipment": (368, 117),
    "organize_the_condiment_bottles": (1442, 0),
    "organize_the_desk": (573, 0),
    "organize_the_hangers": (115, 0),
    "organize_the_makeup": (731, 0),
    "organize_the_medicine_kit": (267, 0),
    "organize_the_mixed_jewelry": (168, 0),
    "organize_the_sunglasses": (434, 0),
    "pack_the_candies": (382, 0),
    "pack_the_chocolate": (539, 0),
    "pack_the_cookies": (464, 0),
    "pack_the_luggage": (511, 57),
    "pack_the_student_bag": (1866, 112),
    "pack_the_takeout_coffee": (1794, 70),
    "pack_the_tool_kit": (225, 112),
    "pack_up_texas_hold_em_set": (407, 0),
    "pack_up_the_badminton_gear": (286, 0),
    "pack_up_the_instrument_practice_props": (263, 0),
    "paint_the_nails": (305, 0),
    "pin_the_brooch_onto_the_clothing": (263, 0),
    "place_and_organize_the_beverage_onto_the_shelf": (222, 0),
    "place_and_organize_the_candy_bags_onto_the_shelf": (142, 0),
    "place_and_organize_the_canned_foods_onto_the_counter": (75, 125),
    "place_and_organize_the_chips_bags_onto_the_shelf": (84, 0),
    "place_and_organize_the_cleaning_sponges_onto_the_shelf": (53, 141),
    "place_and_organize_the_fake_fruits_in_the_fruit_bowl": (193, 622),
    "place_and_organize_the_paper_towels_onto_the_shelf": (201, 0),
    "place_and_organize_the_pasta_boxes_and_bags_onto_the_shelf": (232, 202),
    "place_and_organize_the_plastic_toys_onto_the_shelf": (172, 128),
    "place_and_organize_the_shaving_razors_onto_the_shelf": (126, 0),
    "place_and_organize_the_shoes_in_the_shoe_cabinet": (89, 0),
    "place_and_organize_the_shoes_onto_the_shoe_shelf": (131, 0),
    "place_and_organize_the_stuffed_toys_onto_the_shelf": (193, 0),
    "place_and_organize_the_toothpastes_onto_the_shelf": (70, 0),
    "place_the_beverage_into_the_canvas_bag": (182, 0),
    "place_the_bra_in_the_laundry_bag_and_zip_it_closed": (250, 0),
    "place_the_bread": (1362, 0),
    "place_the_coffee_filter_in_the_dripper": (448, 81),
    "place_the_cup_by_the_coaster": (423, 234),
    "place_the_fake_bread_into_the_paper_bag": (124, 169),
    "place_the_fake_fruits_into_the_plastic_bag_and_tie_a_complex_knot": (197, 41),
    "place_the_flowers_into_the_vase": (493, 0),
    "place_the_food_in_the_zip_top_bag_and_seal_the_zipper": (2573, 119),
    "place_the_food_into_the_grocery_bag_and_tie_it": (32, 104),
    "place_the_fruits_into_the_plastic_bag": (169, 144),
    "place_the_glasses_into_the_tray": (88, 162),
    "place_the_letter_into_the_envelope_and_seal_it": (283, 89),
    "place_the_mixed_dishes_into_the_plastic_bin": (306, 134),
    "place_the_mixed_food_into_the_grocery_bags_and_tie_them": (56, 78),
    "place_the_personal_care_products_into_the_canvas_bag": (225, 0),
    "place_the_phone_on_the_phone_stand": (322, 0),
    "place_the_plates_into_the_plastic_bin_on_the_countertop": (946, 200),
    "place_the_shirt_on_the_hanger": (247, 0),
    "place_the_shorts_on_the_hanger": (298, 0),
    "place_the_skirt_on_the_hanger": (217, 0),
    "place_the_snacks_into_the_paper_bag": (505, 218),
    "place_the_t_shirt_on_the_hanger": (750, 0),
    "place_the_tank_top_on_the_hanger": (341, 0),
    "place_the_trousers_on_the_hanger": (482, 0),
    "place_the_utensils_on_the_paper_napkin_and_roll_it_up": (206, 77),
    "place_the_wine_glass_upside_down_on_the_wine_glass_rack": (440, 0),
    "prepare_the_surgical_pack": (129, 0),
    "prepare_toiletry_sets_of_a_toothbrush_and_toothpaste_on_the_tray": (200, 0),
    "pull_the_plug_from_the_switch_port": (327, 0),
    "pull_the_plug_off_the_socket": (955, 52),
    "put_away_the_umbrella": (216, 0),
    "put_the_credit_cards_into_the_card_holder": (2502, 72),
    "put_the_files_into_the_folder": (343, 170),
    "put_the_keys_on_the_keyring": (2537, 268),
    "put_the_phone_into_the_phone_case": (443, 141),
    "put_the_photo_into_the_frame": (874, 24),
    "put_the_pillow_into_the_pillowcase": (536, 0),
    "put_the_plastic_bottles_in_the_bin": (3768, 25),
    "put_the_screwdriver_in_the_bin": (1004, 1230),
    "put_the_trash_bags_into_the_trash_bin": (521, 0),
    "put_the_wine_bottle_on_the_wine_rack": (214, 0),
    "remove_the_brooch_from_the_garment": (341, 0),
    "remove_the_keys_from_the_keyring": (598, 147),
    "remove_the_microphone_from_the_stand": (342, 0),
    "remove_the_phone_from_the_phone_stand": (279, 202),
    "remove_the_pillowcase_from_the_pillow": (664, 0),
    "remove_the_shirt_from_the_hanger": (212, 0),
    "remove_the_shorts_from_the_hanger": (384, 0),
    "remove_the_skirt_from_the_hanger": (297, 0),
    "remove_the_socks_clipped_to_the_hanger": (345, 0),
    "remove_the_t_shirt_from_the_hanger": (394, 0),
    "remove_the_tank_top_from_the_hanger": (483, 0),
    "remove_the_trousers_from_the_hanger": (439, 0),
    "remove_the_underwear_clipped_to_the_hanger": (136, 0),
    "roll_out_the_dumpling_wrappers_from_the_dough": (133, 0),
    "roll_the_socks": (532, 54),
    "roll_the_t_shirts": (271, 0),
    "roll_the_ties": (548, 0),
    "roll_the_towels": (585, 0),
    "roll_the_underwear": (378, 0),
    "roll_up_the_long_pants": (107, 0),
    "screw_on_the_bottle_caps": (370, 0),
    "serve_the_afternoon_tea": (229, 0),
    "serve_the_lunch_box": (407, 0),
    "serve_the_pet_food": (229, 0),
    "set_the_dinner_table_for_one_person": (1613, 0),
    "set_up_the_chess_pieces_on_the_board": (1102, 86),
    "set_up_the_instrument_practice_props": (204, 0),
    "set_up_the_pet_bed": (110, 0),
    "set_up_the_pet_tent": (98, 0),
    "snap_on_the_fuse_box": (75, 0),
    "sort_the_eating_utensils_into_containers": (536, 0),
    "sort_the_hair_cutting_tools": (154, 0),
    "sort_the_legos_into_containers_by_color": (4070, 388),
    "sort_the_pills_into_containers": (501, 9),
    "sort_the_screws_and_nuts_into_containers": (18, 132),
    "sort_the_stationery_into_containers": (2079, 0),
    "sort_the_tools_into_containers": (609, 0),
    "stack_the_books_and_files": (359, 158),
    "stack_the_hats": (293, 0),
    "sweep_away_the_paper_scraps_from_the_table": (378, 105),
    "take_the_beverage_out_of_the_canvas_bag": (201, 0),
    "take_the_credit_cards_out_of_the_card_holder": (2648, 84),
    "take_the_fake_bread_out_of_the_paper_bag": (174, 79),
    "take_the_fake_fruits_out_of_the_plastic_bag": (678, 175),
    "take_the_personal_care_products_out_of_the_canvas_bag": (213, 0),
    "take_the_phone_out_of_the_phone_case": (643, 91),
    "take_the_photo_out_of_the_frame": (257, 0),
    "take_the_snacks_out_of_the_paper_bag": (481, 162),
    "take_the_wine_bottle_off_the_wine_rack": (211, 0),
    "take_the_wine_glass_off_the_wine_glass_rack": (607, 173),
    "throw_the_plastic_bottles_in_the_bin": (26, 407),
    "tie_the_bouquet_of_fake_flowers": (365, 43),
    "tie_the_shoes": (413, 36),
    "turn_the_large_container_upside_down": (334, 0),
    "turn_the_mug_right_side_up": (469, 191),
    "unload_the_batteries_from_the_remote_control": (249, 158),
    "unload_the_bowls_from_the_dish_rack": (320, 0),
    "unload_the_cups_from_the_dish_rack": (402, 0),
    "unload_the_mixed_dishes_from_the_dish_rack": (405, 100),
    "unload_the_plates_from_the_dish_rack": (1191, 142),
    "unload_the_staples_from_the_stapler": (344, 0),
    "unlock_the_padlock_with_the_key": (1844, 0),
    "unlock_with_the_key": (473, 154),
    "unpack_the_express_box": (245, 15),
    "unscrew_the_bottle_caps": (294, 0),
    "untangle_the_cables": (2782, 168),
    "untie_the_knot_of_the_grocery_bags_and_take_out_the_mixed_food": (73, 0),
    "untie_the_knot_of_the_plastic_bag_and_take_out_the_fruit": (176, 153),
    "unzip_the_laundry_bag_and_remove_the_bra": (770, 0),
    "wrap_the_cables": (324, 80),
    "wrap_the_gift_box_in_paper": (33, 72),
    "wrap_the_headphones": (1150, 154),
    "write_a_sentence_on_the_whiteboard": (1394, 0),
    "zip_tie_the_wires": (57, 0),
    "zip_up_the_jacket": (751, 122),
}
"""Task → (realsense, zedx) episodes, ``ABC_SKIP`` excluded."""

ABC_SUBSET: dict[str, dict[str, tuple[str, ...]]] = {
    "abc_yam_realsense": {
        "load_the_mixed_dishes_into_the_dish_rack": ("a0f20960-e24e-4e16-8c13-156bb47318bf",),
        "place_and_organize_the_fake_fruits_in_the_fruit_bowl": (
            "0013e3e2-89bf-4462-8eca-f05db3233edd",
            "00bef510-59fc-452f-b003-e090bae6367a",
            "0523dbf7-d52a-4b08-918f-420a7812a0b6",
            "063a79f6-d05c-476d-b585-f4306a374063",
            "07a033eb-e69a-43ad-8eba-6f9f30c49e08",
            "0a33a127-446f-4bcc-af9b-7b620bed8396",
            "0a4c025d-f075-404d-a4a8-ecf06b89bf7b",
            "0accd09b-e7d0-4353-ae46-144f5ef8405e",
            "0b9c3251-3a5d-4399-aac5-17ddfe05be1f",
            "0bf1d623-0b91-4c4e-ad96-1db6dd3e3e92",
            "0d66b6bf-4836-47d3-ae1c-0d31f1a8fe0a",
            "0eee09d9-9960-414e-aafb-9581d716a60b",
            "1183c3d1-25b2-4334-a273-cf44d02bfb29",
            "11d09b46-cf9b-401f-b2d1-e1320cf92857",
            "1294dbe7-2d8b-4043-8be6-0777b4d3f3dc",
            "12b5a440-00a8-4148-8330-dfbd82acf2cd",
            "12df5b26-b516-4941-8ae4-18038365e219",
            "17d8bd7e-6881-4cad-b865-7ba2cf369c00",
            "18e71589-a063-434c-adee-e34b925f9bea",
            "1ce9f9cb-94c9-404e-9cec-a407fbb12d28",
            "1cedc64a-db74-4d7c-bc68-1c39e19d01d4",
            "2061245c-4526-4128-b29b-5f3907282d71",
            "2103b6ec-d299-442e-a30f-97d64d7535ec",
            "21977e63-c292-43b6-b26b-8ca45cb573d9",
            "235093ea-5496-41eb-889b-9e2da54b311c",
            "2435ad70-d12d-40a3-a57d-27c9a120ff41",
            "25308a45-1e7d-4af8-8528-1d77b4d02941",
            "25441dd9-1dc6-40de-89cd-da21a2b9ce43",
            "259c4a43-344a-4045-9916-6aab4ce876e1",
            "26bed694-9452-4ff0-a83e-165a0b4f8ed8",
            "2bf0afe4-bc91-42fa-b75f-71a2c8d3d2c5",
            "2e1b698a-c8ee-4ddc-ac00-6c7f168bc761",
            "2ecf5ca3-a02e-436d-a147-ad17b80f36ca",
            "2f73d078-8c4e-413e-a334-de440feb52a6",
            "30529148-fd29-44f8-add9-75c7a08c7d93",
            "318f860b-43ae-402e-8db3-465bf6382bbf",
            "321b9ff0-ca14-4a58-b21c-a618854bfbaf",
            "323f9065-b7ee-48d5-bc9a-8317911f60fc",
            "345b937a-765e-454e-81d7-4783b1731fe5",
            "3508922e-df08-4ac0-a625-b88e7dfea239",
            "37f6af0c-fed8-4636-9ac2-a5787c008d0a",
            "38fadca2-f93f-412d-abe9-8a48932dee62",
            "3a57ff78-77f4-4148-adad-c2c613725860",
            "3e970151-7974-4289-a6e0-d88bfde69b9b",
            "3f7f0a70-78fa-44f1-ba9a-ecde123a551b",
            "3fa2f44a-269f-447a-9968-af7a6d3ca91d",
            "408ba284-7cff-45ca-b534-34b7be87fac7",
            "4155d182-1dfe-4d95-b793-0b91d84219d8",
            "418f3b8e-2eff-4194-8803-dfb98c509e51",
            "42508817-7a98-4a67-82d0-fa34eea5f52a",
            "4383d6fa-e936-40cd-9296-31f7f61d7423",
            "44204a6f-b197-43c6-a566-82e504a573c6",
            "4596785e-6a81-4e9a-8795-c41a055ebee2",
            "47b11b5f-b26d-4994-92ba-7e0b59c81229",
            "49719908-33cd-4960-b48b-c6822a61021a",
            "4b9404bf-492c-4808-bb09-cf8cbd8ac988",
            "4c10d7e8-f02a-425b-a050-eaec73eda422",
            "4f54a713-8ce2-459b-82b6-eb68189acf75",
            "4ffb848f-107c-4a8a-aa43-e3ac6d6e58a4",
            "51508222-f680-401d-a8dd-9e4c93068d9a",
            "524e2353-0f49-46a3-b408-595b1eb74109",
            "54ab92a1-9573-401d-89cc-c145ff5f0175",
            "5a0847a4-6c34-44b6-8c02-adcbe1a748dc",
            "5e7c490f-d418-42e3-bb46-689affb96633",
            "5ea0180d-bbbc-4e70-8073-d5add89514e2",
            "5fe2eff1-6f80-4199-9ef4-fe8f545c4e25",
            "61d8936b-d13b-4529-aa36-d53e4d004f12",
            "6201068a-358d-4fc2-882a-f515a042f5e6",
            "626fd9c0-1f72-49ce-bf16-5027f3c6e884",
            "62a92cd1-8e6e-4580-8c82-d24e9079e7ef",
            "62b3aaf5-69e0-430c-98bf-20714657fed9",
            "638fc361-d611-40b3-8e1e-ca386b202b8a",
            "65e6f030-12ca-4890-a73f-13baeffeeade",
            "67e62603-c1a4-400e-af79-b13d014c05cf",
            "68855b82-570b-442f-a06d-0711aca0b725",
            "691ae56a-ab1b-4872-986c-381a7b219495",
            "69ae92b7-8f4f-4501-b6ec-d5aadf3ee95b",
            "6b70c609-9703-4f06-a88e-d9aeb7680d59",
            "6bdb1193-a088-4352-9f5b-587464b92501",
            "6bf1cf66-a37c-4760-be8d-d392fa4011fd",
            "6e520255-7cd4-424a-a73d-49d60648ad38",
            "6e793d17-4b19-4b12-a874-6ac6bf166fcf",
            "6f7cea17-72c6-4838-aa5b-5d8dc2d76876",
            "711df355-89d4-457b-b237-f558f7a61298",
            "714e031a-5fa7-4164-abff-8e5340b95f3c",
            "7506c77b-661c-4c58-9f40-2f7d0a74c9e2",
            "75fc648a-22e5-426d-9cab-1a26d4f0fcd9",
            "77046dee-0f65-4264-a534-ed406036a19f",
            "7b77cc3a-912e-45c1-9d3c-795b66902fe9",
            "7bd498ad-f498-4253-a0c6-6ec7d1080230",
            "7d6381d7-96e1-43b2-a67c-7056961aadae",
            "7deb9579-b906-4578-8d65-c0df55413fc1",
            "7ec89431-4b44-4573-a8d3-51d26e26691e",
            "7ecae31d-7d38-48bd-963c-a11c8b960e1b",
            "7edc3f85-e007-4977-9330-95116ed0fcb4",
            "7f249316-c3fe-41db-86c9-fa16a7955e58",
            "80166c93-0bba-4644-8058-699b686d8f18",
            "8081c34a-71a2-4717-96c9-96ea706cc152",
            "808b0fc2-a071-403f-98da-643c85f8f24f",
            "84405a89-e1dc-4e69-8ff2-736b984f5296",
            "8547a540-f1ae-40da-8d4c-6ef17628b59e",
            "867dbc0d-21c9-4def-8b4b-87e7b4e9a7c6",
            "86fc5acd-5a04-4254-9118-4e52ec224fdb",
            "88be208c-d3da-4c81-b2eb-ce429faeee45",
            "8968ef41-6fc1-4b5c-9bd3-1c42da4db5f0",
            "8af78b3c-b5dd-418b-b63e-3cc6b520b3c7",
            "8e9837c6-5280-471e-bf5c-73d9c5ee3fdb",
            "8eaafd73-d001-47ac-87e4-57555d68a082",
            "9167db86-ab93-48ac-905b-0aa9b389ef64",
            "92986ed7-1190-44d7-8f09-186b32ac0ed3",
            "94508d05-a05d-4176-9661-3296607d7e75",
            "9644bf69-d8af-48b5-894c-89144eb0d66d",
            "97735ac4-e189-4d5b-8c49-801c691cee4e",
            "979b4407-4241-476a-9310-1c2484180530",
            "97d76810-c4e8-4e4f-b82d-0abd2aedbccd",
            "9a0de60e-995a-478f-aa92-5df1e11c0a2f",
            "9a23b654-5be8-443b-bf66-c00d9a297a08",
            "9d1e7f72-944c-4599-82ff-6c698fca5d5c",
            "9d735f43-939f-442a-a001-d779ccc5080e",
            "9e23047f-9088-4b46-bde6-8eeafd911aaf",
            "9e4c2f0d-b1e1-4f72-92f2-96370fa31270",
            "a0019f1d-0812-4770-9834-a08c98d21d01",
            "a0304d50-7a2a-48b3-9976-63ee9f4696ef",
            "a2e6669d-7c40-45a5-ad72-745a49aa7ffe",
            "a476d098-945d-4f0e-b123-0672b97aa598",
            "a4aca32e-5c6f-4226-ac52-643b92a32ff9",
            "a5963365-d004-4ede-9522-6b7c70acf335",
            "a5f85939-4b13-404f-8fef-c60d3d307365",
            "a712c41e-e506-4b88-8b63-98e283b80c84",
            "a7bc778b-1ce0-467d-ba18-70350b1a5615",
            "a8f0daba-f0fb-457d-ad28-d5cf38251154",
            "ab0908a7-c444-4456-a2bc-3357537d7c9a",
            "ae0fc4e3-081e-4b3d-872e-69dfe6fc98c3",
            "afa1294f-67c0-4ac2-b980-44de9131cb2c",
            "b1a0697c-caff-4d35-995f-1c1fa2bb440f",
            "b2d34144-97ff-4c1e-8f06-708aacdb9f0a",
            "b48c227f-7168-47dc-8aa2-918d281a3149",
            "b64d83a6-f4e7-4796-bba7-4016cce40316",
            "b69a8c9d-0eaf-42e8-a794-171242b67211",
            "b76fc1c4-bd79-4b1f-82a9-563b5d6b0b57",
            "b899aeeb-f50d-40d2-ac18-d6258a06db7a",
            "b93f350d-f036-4159-aa80-df6fc018cc7c",
            "b9479ccb-683a-4e74-9187-a2a0b8f74f01",
            "b98bd098-10f6-4a9a-bfcc-6100fc1f2966",
            "bccb23b2-b87d-4774-ad39-a2f87d46c948",
            "c0ea5dc5-9b7e-48bd-b639-e2d6c8742518",
            "c15b366b-7d05-489a-99ac-e42f7ab8fa50",
            "c1c1a1aa-0265-4c4d-9df3-bf8ff0c286ea",
            "c40e4b78-aa4d-4903-8484-fe13355b62f8",
            "c6e34394-eb8d-4f4e-b8d5-3decefeb663d",
            "c6e5f31a-a5bb-475a-8569-cfe895314190",
            "c772067a-1317-4d52-848c-adca4b17f71a",
            "c84812c4-646e-4eb6-9f75-1e69eb3fb757",
            "c99fccaf-8153-48cd-b45f-89399360b33e",
            "cacde17b-002b-4a40-9d2d-b5cf94b46631",
            "cad786bc-a442-486b-84fc-ad85216856ea",
            "cc6eb754-4b1a-412a-bf96-61057075695e",
            "ce722c23-8c5b-40c1-9363-1d67bb314119",
            "ceb0fc07-1269-4172-baf2-99d677e8647b",
            "d327201c-b351-4c52-a60a-6f960d523986",
            "d36e6068-14b2-4b15-8685-7dbd6d1d4a83",
            "d5128bef-aa4e-45e9-8390-9597a611c0c3",
            "d5e0adbc-3807-46c7-91eb-952c50f752a3",
            "d6f17344-fbdc-4b7b-9445-b3d3ba9f109c",
            "d7a34560-3309-4b4a-96d9-8550d0ba8d3b",
            "d898aeba-2cbd-439f-bcea-75477f4a5689",
            "d8d1ac49-f372-49fc-bc65-c3f8042e4215",
            "da218aeb-a605-4f5d-8a09-1e2cbab069cb",
            "daff8c91-5643-4955-9be7-ef28f1e636ab",
            "dd4d65fa-19ed-4f0c-bc96-92fe74657b82",
            "e3a6f9af-55a7-4d1d-b656-d56a52d00fcf",
            "e41842ec-fc62-4efc-9f5a-2edd21ff2ce3",
            "e483045c-2358-427d-912b-40888abd6724",
            "e593da0a-6370-41b5-95da-5f59325db398",
            "e772ce32-b434-42f9-b349-266a0096f80e",
            "e7de091e-5c47-4b64-b7d6-62fd466d034d",
            "e840a86a-6c8b-4a3b-934a-000d613a01cd",
            "e8920da2-303d-45ca-9699-dc3e3b16928c",
            "e9c68503-1645-4f8e-9cb7-98ced7da771a",
            "ea4fb9a1-b7e7-44a9-b382-12b17342af37",
            "eba0bf72-bb45-40f8-9da0-203f98c26db9",
            "ed3a5200-37d2-4634-afe6-f1f9795cae50",
            "f0cfadde-263a-419d-a5a8-344bf7c223ab",
            "f2aad278-d52d-4cb6-aacd-a4bc82383608",
            "f2c74e34-6b82-42e0-98e9-247ffddae8c0",
            "f570c7be-f848-43e8-a168-150bf3484f4b",
            "f7176c07-4b8c-4150-9772-7a2e0ad1e54a",
            "f91acde3-0783-44b9-9383-417918acfa29",
            "fa67703f-a530-4e25-ac8c-bbda9c0c15c1",
            "fbeba763-4993-4ea8-9adb-8cb9c085b1cf",
            "fe8db738-b4fe-4ebf-8de8-e1c5cb9be1f2",
            "fef5576a-e138-4e21-8443-91f9935c61cd",
            "ffd59f4f-4c0a-4365-b02f-f95680ea8202",
        ),
        "put_the_screwdriver_in_the_bin": ("c57eb604-038b-4c10-9da1-028403a9abe8",),
        "sort_the_eating_utensils_into_containers": ("13a31c92-dc56-423d-b0f8-8c8692c5d575",),
        "untangle_the_cables": ("db840528-84b8-45da-82d8-3727a599ba41",),
        "wrap_the_cables": ("73ed6bbf-51ba-4037-9c96-631dfd9dab24",),
        "write_a_sentence_on_the_whiteboard": ("cf4440ce-0c37-4a60-9e76-a491b6cda96b",),
    },
    "abc_yam_zedx": {
        "load_the_plates_into_the_dish_rack": ("74de0979-3d4b-4d6e-b562-810bcc2fb3a2",),
        "sort_the_pills_into_containers": (
            "3e8596cb-cdea-44de-a74f-7e5852c3c717",
            "50285f25-d493-4960-bfae-51534f72a10c",
            "53150c38-632c-4a99-b1a4-4e2eebae8c39",
            "86c788bd-e11f-4499-b13a-7f1405673324",
            "aaeee028-80fa-46de-8314-f06960433781",
            "d7174805-1bc1-488b-be2d-53e331460514",
            "e316ff1b-2192-4819-8034-f8c710479fa8",
            "ef701f2b-7131-4275-9631-496d91d3110c",
            "f4b02c7e-d348-4ea5-b76f-2bcd45e8b096",
        ),
    },
}
"""``--subset`` episodes per station: synced 22/27/30 Hz and async 30 (zedx)/43/60 Hz, a synced
episode with mistakes, the realsense leaf of one task and the zedx leaf of another."""


def _pin(allow: tuple[str, ...], episodes: int) -> Pin:
    source = HfSnapshot("XDOF/ABC-130k", ABC_REVISION, allow)
    return Pin((source,), episodes=episodes, frames=None)


PINS = {
    name: replace(
        _pin(
            tuple(f"data/train/{t}/*" for t, n in ABC_TASKS.items() if n[col]),
            sum(n[col] for n in ABC_TASKS.values()),
        ),
        subset=_pin(
            tuple(
                f"data/train/{t}/episode_{e}/*" for t, eps in ABC_SUBSET[name].items() for e in eps
            ),
            sum(map(len, ABC_SUBSET[name].values())),
        ),
    )
    for col, name in enumerate(STATIONS)
}
"""``frames=None``: an ABC frame total is the sum of the 30 Hz grid lengths, unknown before a
build; the build checks each episode (grid length = rows = video frames)."""

RECIPES = {
    name: Recipe(Abc(), curate=(), stats=(op("global_qk", pool=tuple(STATIONS)),))
    for name in STATIONS
}
