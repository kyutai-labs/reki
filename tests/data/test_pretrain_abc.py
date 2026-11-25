"""ABC-130k 30 Hz resample rule (``pretrain.readers.abc``) on synthetic MCAP timestamps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pretrain.readers import abc

pytestmark = pytest.mark.cpu

T0 = 1_775_013_591_123_477_000
MS = 1_000_000


def _clock(hz: float, n: int, start: int = T0, drop: tuple[int, ...] = ()) -> np.ndarray:
    ts = start + np.floor(np.arange(n) * 1e9 / hz).astype(np.int64)
    return np.delete(ts, list(drop))


def test_grid_is_real_time_30hz_over_the_common_span() -> None:
    top = _clock(60, 600, T0 + 5 * MS)
    state = _clock(270, 2700, T0)
    action = _clock(200, 1900, T0 + 2 * MS)
    ticks = abc.tick_grid(
        [s[0] for s in (top, state, action)], [s[-1] for s in (top, state, action)]
    )
    assert ticks[0] == T0 + 5 * MS
    assert ticks[-1] <= min(top[-1], state[-1], action[-1]) < ticks[-1] + 1e9 / 30
    np.testing.assert_array_equal(ticks - ticks[0], np.arange(len(ticks)) * 10**9 // 30)


def test_disjoint_streams_raise() -> None:
    with pytest.raises(ValueError, match="overlap"):
        abc.tick_grid([0, 100], [50, 200])


def test_camera_floor_30hz_with_drops() -> None:
    cam = _clock(30, 90, drop=(10, 11, 12, 40))
    ticks = abc.tick_grid([cam[0]], [cam[-1]])
    idx = abc.floor_indices(cam, ticks)
    assert (cam[idx] <= ticks).all()
    assert ((idx + 1 >= len(cam)) | (cam[np.minimum(idx + 1, len(cam) - 1)] > ticks)).all()
    # the 3-frame hole repeats the frame before it 3 times, the 1-frame hole once
    assert (np.diff(idx) == 0).sum() == 4
    assert len(ticks) == 90


def test_camera_floor_60hz_takes_every_other_frame() -> None:
    cam = _clock(60, 601)
    ticks = abc.tick_grid([cam[0]], [cam[-1]])
    idx = abc.floor_indices(cam, ticks)
    assert len(ticks) == 301
    np.testing.assert_array_equal(idx, 2 * np.arange(301))


def test_camera_floor_43hz_60hz_clock_with_holes() -> None:
    cam = _clock(60, 900, drop=tuple(range(3, 900, 4)))
    ticks = abc.tick_grid([cam[0]], [cam[-1]])
    idx = abc.floor_indices(cam, ticks)
    assert (cam[idx] <= ticks).all() and (ticks - cam[idx] < 2 * 1e9 / 60).all()


def test_tick_before_first_frame_raises() -> None:
    with pytest.raises(ValueError, match="precedes"):
        abc.floor_indices(np.array([10, 20]), np.array([5, 15]))


def test_interpolation_is_linear_in_time() -> None:
    ts = _clock(200, 400)
    values = np.stack([(ts - T0) / 1e9, -3 * (ts - T0) / 1e9 + 1], 1)
    ticks = abc.tick_grid([ts[0]], [ts[-1]])
    out = abc.interpolate(ts, values, ticks)
    assert out.dtype == np.float32
    expected = np.stack([(ticks - T0) / 1e9, -3 * (ticks - T0) / 1e9 + 1], 1)
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_synced_episode_with_holes_has_no_repeated_rows() -> None:
    """Synced mode: one state row per 22-30 Hz loop tick, with ~100 ms holes. Floor repeats rows
    in the holes; interpolation does not."""
    loop = _clock(30, 300, T0 + 110 * MS, drop=tuple(range(5, 300, 4)))
    values = np.sin((loop - T0) / 1e9)[:, None] + np.zeros((1, 6))
    ticks = abc.tick_grid([loop[0]], [loop[-1]])
    out = abc.interpolate(loop, values, ticks)
    assert (np.abs(np.diff(out, axis=0)).max(1) > 0).all()
    floor = values[abc.floor_indices(loop, ticks)]
    assert (np.abs(np.diff(floor, axis=0)).max(1) == 0).any()


def test_async_270hz_interpolation_close_to_floor() -> None:
    ts = _clock(270, 2700)
    values = np.sin((ts - T0) / 1e9)[:, None]
    ticks = abc.tick_grid([ts[0]], [ts[-1]])
    out = abc.interpolate(ts, values, ticks)
    floor = values[abc.floor_indices(ts, ticks)]
    assert np.abs(out - floor).max() < 5e-3


def test_interpolation_outside_span_raises() -> None:
    with pytest.raises(ValueError, match="span"):
        abc.interpolate(np.array([10, 20]), np.zeros((2, 1)), np.array([10, 25]))


def test_subtask_intervals_on_ticks() -> None:
    ticks = T0 + np.arange(100, dtype=np.int64) * 10**9 // 30
    anns = [
        (ticks[10] + 1, "Pick up the cup."),
        (T0 - 5 * MS, "Reach."),
        (ticks[30], "Place the cup."),
        (ticks[50] - 1, "mistake"),
        (ticks[60], "Place the cup."),
        (ticks[60] + 2, "adjust"),
        (ticks[-1] + 5 * MS, "late"),
    ]
    intervals, mistakes = abc.subtask_intervals(anns, ticks)
    assert intervals == [
        [0, 11, "Reach."],
        [11, 30, "Pick up the cup."],
        [30, 50, "Place the cup."],
        [60, 61, "Place the cup."],
        [61, 100, "adjust"],
    ]
    assert mistakes == [2]


def test_no_annotations() -> None:
    assert abc.subtask_intervals([], np.arange(5, dtype=np.int64)) == ([], [])


def test_task_text() -> None:
    assert abc.task_text("wrap_the_cables") == "Wrap the cables."


def test_pins() -> None:
    assert len(abc.ABC_TASKS) == 197 and len(abc.ABC_SKIP) == 4
    assert abc.PINS["abc_yam_realsense"].episodes == 116_462
    assert abc.PINS["abc_yam_zedx"].episodes == 12_566
    for name, pin in abc.PINS.items():
        col = list(abc.STATIONS).index(name)
        assert all(abc.ABC_TASKS[t][col] for t in abc.pinned_episodes(pin))
        assert pin.subset is not None
        sub = abc.pinned_episodes(pin.subset)
        assert sum(len(v or ()) for v in sub.values()) == pin.subset.episodes
        assert all(abc.ABC_TASKS[t][col] for t in sub)


def test_pinned_episodes() -> None:
    pin = abc._pin(("data/train/b/*", "data/train/a/episode_2/*", "data/train/a/episode_1/*"), 3)
    assert abc.pinned_episodes(pin) == {"a": ["episode_1", "episode_2"], "b": None}


def _upstream(tmp_path: Path, eps: dict[str, int]) -> tuple[abc.Pin, list[tuple[str, str]]]:
    """Episode dirs of each task under the pinned upstream; the pin and every (task, episode)."""
    names = [(t, f"episode_{t}{k}") for t, n in eps.items() for k in range(n)]
    pin = abc._pin(tuple(f"data/train/{t}/{e}/*" for t, e in names), len(names))
    for t, e in names:
        ep = abc.upstream_dir(tmp_path, pin.sources[0]) / "data" / "train" / t / e
        ep.mkdir(parents=True)
        (ep / "episode.mcap").write_bytes(b"")
    return pin, names


@pytest.mark.parametrize("n", [1, 2, 3, 7, 40])
def test_episode_shards_cover_every_episode_once(tmp_path: Path, n: int) -> None:
    pin, names = _upstream(tmp_path, {"big": 13, "small": 2, "mid": 6})
    shards = [
        [(t, ep.name) for t, ep in abc.Abc().shard_episodes(tmp_path, pin, (i, n))]
        for i in range(n)
    ]
    assert sorted(x for s in shards for x in s) == sorted(names)


def test_episode_shards_split_one_task(tmp_path: Path) -> None:
    pin, _ = _upstream(tmp_path, {"big": 8})
    sizes = [len(abc.Abc().shard_episodes(tmp_path, pin, (i, 4))) for i in range(4)]
    assert sum(sizes) == 8 and sum(n > 0 for n in sizes) > 1


def test_convert_writes_only_its_shard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pin, names = _upstream(tmp_path, {"a": 3, "b": 4})
    monkeypatch.setattr(abc, "station_of", lambda mcap: "abc_yam_realsense")
    monkeypatch.setattr(
        abc, "convert_episodes", lambda jobs: [tmp.mkdir(parents=True) or 0 for *_, tmp, _ in jobs]
    )
    work = tmp_path / ".work" / "abc" / "abc_yam_realsense"
    done: list[set[str]] = []
    for i in range(3):
        before = {p.name for p in work.glob("*/*")} if work.exists() else set()
        abc.Abc().convert("abc_yam_realsense", tmp_path, pin, (i, 3))
        done.append({p.name for p in work.glob("*/*")} - before)
    assert sum(map(len, done)) == len(names) and set().union(*done) == {e for _, e in names}
    assert not list(work.glob("*/*.part"))


def test_resample_video_needs_an_even_size(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="even size"):
        abc._ffmpeg(
            [([], "h264", np.zeros(1, dtype=np.int64), (481, 640), tmp_path / "x.mp4")], False
        )
