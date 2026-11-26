"""ABC-130k camera encode (CPU or NVDEC decode → nv12 → libx264) on a synthetic stream: each
output frame is its pick's source frame, ≥35 dB, with no brightness shift."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
from pretrain.readers import abc

pytestmark = pytest.mark.gpu

H, W, N = 480, 640, 40


def _rgb(path: Path, fmt: str | None = None) -> np.ndarray:
    import av

    c = av.open(str(path), format=fmt)
    frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    c.close()
    return np.stack(frames).astype(float)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    return float(10 * np.log10(255**2 / max(((a - b) ** 2).mean(), 1e-10)))


@pytest.mark.parametrize(("profile", "nvdec"), [("baseline", False), ("high", True)])
def test_encode_matches_picked_source_frames(tmp_path: Path, profile: str, nvdec: bool) -> None:
    import av

    src = tmp_path / "src.h264"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi"]
        + ["-i", f"testsrc=size={W}x{H}:rate=30", "-frames:v", str(N)]
        + ["-c:v", "libx264", "-profile:v", profile, "-g", "10", "-pix_fmt", "yuv420p"]
        + ["-f", "h264", str(src)],
        check=True,
    )
    c = av.open(str(src), format="h264")
    packets = [bytes(p) for p in c.demux(video=0) if p.size]
    c.close()
    assert len(packets) == N
    picks = np.sort(np.random.default_rng(0).integers(0, N, size=55))
    assert len(set(picks)) < len(picks) and len(set(picks)) < N  # repeats and holes
    with abc._template(packets, "h264") as probe:
        ctx = probe.streams.video[0].codec_context
        assert (
            abc._cpu_decodable("h264", ctx.profile or "", (ctx.height, ctx.width), (H, W)) != nvdec
        )
    out = tmp_path / "out.mp4"
    abc._ffmpeg([(packets, "h264", picks, (H, W), out)], nvdec)

    source, frames = _rgb(src, "h264"), _rgb(out)
    assert len(frames) == len(picks)
    for i, want in enumerate(picks):
        psnr = [_psnr(frames[i], s) for s in source]
        assert int(np.argmax(psnr)) == want
        assert psnr[want] >= 35
    shift = (frames - source[picks]).mean(axis=(0, 1, 2))
    assert np.abs(shift).max() < 0.5
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries"]
        + ["stream=pix_fmt,width,height", "-of", "csv=p=0", str(out)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert probe.stdout.strip() == f"{W},{H},yuv420p"
