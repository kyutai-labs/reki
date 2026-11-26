"""ffprobe queries on the first video stream of an mp4, the frame encoder, and the join of
per-episode mp4s into one file."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np

TICKS_PER_FRAME = 512
"""Time-base ticks per frame of a joined file (ffmpeg's 1/15360 at 30 fps)."""


def ffprobe(path: Path, *args: str) -> str:
    """The stripped stdout of ``ffprobe -v error -select_streams v:0 <args> <path>``."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", *args, str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def video_frames(path: Path) -> int:
    """The number of video frames (packets) of an mp4."""
    return int(
        ffprobe(path, "-count_packets", "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0")
    )


def stream_frames(path: Path) -> int:
    """The frame count in the video stream header of an mp4."""
    import av

    frames = -1
    with av.open(str(path)) as c:
        frames = c.streams.video[0].frames
    return frames


def packet_pts(path: Path) -> tuple[np.ndarray, Fraction, float]:
    """The sorted video packet pts (time-base ticks) of an mp4, its time base and the frame
    rate the training loader reads from its header (torchcodec's ``average_fps_from_header``,
    not ffprobe's ``avg_frame_rate``)."""
    from torchcodec.decoders import VideoDecoder

    tb = Fraction(ffprobe(path, "-show_entries", "stream=time_base", "-of", "csv=p=0"))
    out = ffprobe(path, "-show_entries", "packet=pts", "-of", "csv=p=0")
    fps = VideoDecoder(str(path), seek_mode="approximate").metadata.average_fps_from_header
    if fps is None:
        raise ValueError(f"{path}: no average frame rate in the header")
    return np.sort(np.array(out.split(), dtype=np.int64)), tb, fps


def loader_pts(pts: np.ndarray, time_base: Fraction, fps: float, ts: np.ndarray) -> np.ndarray:
    """The pts (s) of the frames the training loader decodes for the timestamps ``ts`` (s) of
    a file with the sorted packet ``pts``, its ``time_base`` and header frame rate ``fps``; nan
    past the last frame. The loader asks torchcodec (approximate mode) for frame index
    ``round(t * fps)``, which seeks ``round(index / fps / time_base)`` and returns the first
    frame at or after it: a frame one tick early is skipped."""
    index = np.round(ts * fps)
    target = np.round(index / fps / float(time_base))
    pos = np.searchsorted(pts, target, side="left")
    out = np.full(len(ts), np.nan)
    inside = (pos < len(pts)) & (index < len(pts))
    out[inside] = pts[pos[inside]] * float(time_base)
    return out


def stream_format(path: Path) -> tuple[tuple, int]:
    """The format (codec, width, height, pixel format) of the video stream of an mp4 and its
    frame count."""
    import av

    fmt: tuple = ()
    frames = 0
    with av.open(str(path)) as container:
        s = container.streams.video[0]
        fmt = (s.codec_context.name, s.width, s.height, s.codec_context.pix_fmt)
        frames = s.frames
    return fmt, frames


def join_mp4s(srcs: list[Path], dst: Path, fps: int) -> None:
    """Join the mp4s ``srcs`` (one stream format) into ``dst`` without re-encoding. Frame ``k``
    of ``dst`` sits at exactly ``k / fps``: each source's frames are placed on the frame grid
    from their pts, so no rounding builds up at the joins (ffmpeg's concat demuxer offsets by
    durations in microseconds and drifts by a tick). Raises when a source's frames are off
    its own ``fps`` grid (missing, repeated or more than half a frame early or late)."""
    import av

    time_base = Fraction(1, fps * TICKS_PER_FRAME)
    files: list[tuple[list, list[int]]] = []
    delay = 0
    for src in srcs:
        with av.open(str(src)) as container:
            stream = container.streams.video[0]
            tb = stream.time_base
            assert tb is not None
            timed = [(p, t) for p in container.demux(stream) if (t := p.pts) is not None]
            if len(timed) != stream.frames:
                raise ValueError(f"{src}: {len(timed)} packets, header {stream.frames} frames")
            packets = [p for p, _ in timed]
            first = min(t for _, t in timed)
            pos = [round((t - first) * tb * fps) for _, t in timed]
            if sorted(pos) != list(range(len(pos))):
                raise ValueError(f"{src}: frames off the {fps} fps grid")
            # Decode order i shows frame pos[i]; dts = i - delay stays at or before every pts.
            delay = max(delay, *(i - j for i, j in enumerate(pos)))
            files.append((packets, pos))
    with tempfile.TemporaryDirectory(dir=dst.parent) as tmp:
        out_path = Path(tmp) / dst.name
        with (
            av.open(str(srcs[0])) as template,
            av.open(str(out_path), "w", options={"movflags": "faststart"}) as out,
        ):
            stream = out.add_stream_from_template(template.streams.video[0], opaque=True)
            stream.time_base = time_base
            k0 = 0
            for packets, pos in files:
                for i, (packet, j) in enumerate(zip(packets, pos)):
                    packet.stream = stream
                    packet.time_base = time_base
                    packet.pts = (k0 + j) * TICKS_PER_FRAME
                    packet.dts = (k0 + i - delay) * TICKS_PER_FRAME
                    packet.duration = TICKS_PER_FRAME
                    out.mux(packet)
                k0 += len(pos)
        shutil.move(out_path, dst)


class FrameEncoder:
    """An ffmpeg process that encodes RGB frames (``[H, W, 3]`` uint8, written one by one) into
    the mp4 ``path`` at ``fps`` with the encoder ``args``."""

    def __init__(self, path: Path, fps: int, height: int, width: int, args: tuple[str, ...]):
        self.path = path
        self.size = (height, width)
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24"]
        cmd += ["-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0", *args, "-an", str(path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != (*self.size, 3):
            raise ValueError(f"{self.path}: frame {frame.shape}, stream {self.size}")
        assert self.proc.stdin is not None
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        assert self.proc.stdin is not None and self.proc.stderr is not None
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {self.path}: {err}")

    def kill(self) -> None:
        self.proc.kill()
        self.proc.wait()
