"""AgiBot, Galaxea and RDT readers of the pretraining pipeline, and the shared text rules and
segment lookups (``pretrain.common.{tasks,lookup}``). Synthetic fixtures, no data."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
from pretrain.common import lookup
from pretrain.common.tasks import agibot_subtask, agibot_task, galaxea_text, text_rules
from pretrain.readers import agibot, galaxea, rdt

pytestmark = pytest.mark.cpu


# ------------------------------------------------------------------ text rules


def test_agibot_subtask_verbs_to_present() -> None:
    assert agibot_subtask("The left arm picked up the cup.") == "The left arm picks up the cup."
    assert agibot_subtask("The right arm placed it, then threw the ball.") == (
        "The right arm places it, then throws the ball."
    )
    assert (
        agibot_subtask("The right arm pushed open the door.")
        == "The right arm pushes open the door."
    )


def test_agibot_subtask_keeps_adjectives() -> None:
    for text in (
        "The right arm places the picked-up king beans into the cart.",
        "The right arm places the picked peas into the cart.",
        "The left arm places the cup it was holding.",
        "Waist rotation.",
    ):
        assert agibot_subtask(text) == text


def test_agibot_task_exact_strings_only() -> None:
    assert agibot_task("Handing out flyers to customers") == "Hand out flyers to customers."
    assert agibot_task("Adjust the King Beans to the correct position on the shelf.") == (
        "Adjust the King Beans to the correct position on the shelf."
    )


def test_galaxea_text() -> None:
    assert galaxea_text("organize toys") == "Organize toys."
    assert galaxea_text("pick up the cup ,then place it") == "Pick up the cup, then place it."
    assert galaxea_text("clean the table_20250703") == "Clean the table."
    assert galaxea_text("wipe  the desk 1234567") == "Wipe the desk."
    assert galaxea_text("Done!") == "Done!"
    curtain = "Move forward and look down to the front of the curtain."
    assert galaxea_text("俯视到窗帘前") == curtain == galaxea_text("俯视到窗帘前.")


def test_text_rules_default_identity() -> None:
    rules = text_rules("rdt")
    assert rules.task("a b") == "a b" and rules.subtask("x") == "x"


# ------------------------------------------------------------------ lookup


def _seg(start: int, end: int, text: str) -> dict:
    return {"instruction": text, "start": start, "end": end}


def test_from_segments_drops_episode_wide_and_resolves_overlaps() -> None:
    segs = [_seg(0, 100, "whole"), _seg(0, 40, "a"), _seg(10, 20, "short"), _seg(40, 100, "b")]
    assert lookup.from_segments(segs) == [[10, 20, "short"], [40, 100, "b"]]
    assert lookup.from_segments([_seg(0, 30, "a"), _seg(20, 60, "long"), _seg(60, 100, "c")]) == [
        [0, 30, "a"],
        [60, 100, "c"],
    ]
    assert lookup.from_segments([]) == []


def test_write_lookup_applies_rules_and_skips_empty(tmp_path: Path) -> None:
    (tmp_path / "meta").mkdir()
    lookup.write_segments(
        tmp_path,
        [[_seg(0, 50, "The arm picked it."), _seg(50, 100, "Done.")], [], [_seg(0, 10, "x")]],
    )
    assert lookup.write_lookup(tmp_path, agibot_subtask) == 1
    (line,) = (tmp_path / "meta" / "subtask_lookup.jsonl").read_text().splitlines()
    assert json.loads(line) == {
        "episode_index": 0,
        "intervals": [[0, 50, "The arm picks it."], [50, 100, "Done."]],
    }


# ------------------------------------------------------------------ agibot


def test_task_frames_in_recording_then_frame_order() -> None:
    def frame(kind: str, start: int, comment: str) -> dict:
        detail = {"comment": comment}
        return {"frame_type_name": kind, "start": start, "end": start + 5, "frame_detail": detail}

    info = {
        "key_frame": {
            "10": {"dual": [frame("Task Frame", 50, "b"), frame("Task Frame", 0, "a")]},
            "2": {"dual": [frame("Error Frame", 0, "x"), frame("Task Frame", 3, "c")]},
        }
    }
    assert [(f["episode"], f["start"], f["comment"]) for f in agibot.task_frames(info)] == [
        (2, 3, "c"),
        (10, 0, "a"),
        (10, 50, "b"),
    ]


def test_scatter_state_fills_missing_fields_with_zeros() -> None:
    fields, offset = {}, 0
    for name, dim in agibot.STATE_FIELDS:
        if name.startswith("state/robot/"):
            fields[name] = {"indices": []}
            continue
        fields[name] = {"indices": list(range(offset, offset + dim))}
        offset += dim
    assert offset == 162
    state = np.arange(2 * 162, dtype=np.float32).reshape(2, 162)
    out = agibot.scatter_state(
        state, {"features": {"observation.state": {"field_descriptions": fields}}}
    )
    assert out.shape == (2, 169)
    robot = slice(90, 97)
    assert not out[:, robot].any()
    np.testing.assert_array_equal(np.delete(out, np.r_[robot], axis=1), state)


def test_rescale_only_at_exactly_twice() -> None:
    assert not agibot.rescale((640, 528), (640, 528))
    assert agibot.rescale((1280, 1056), (640, 528))
    with pytest.raises(ValueError):
        agibot.rescale((1920, 1080), (640, 528))


def test_clip_segments_to_the_task_frame() -> None:
    info = {
        "instruction_segments": {
            "4": [
                {
                    "instruction": "a",
                    "skill": "Pick",
                    "start_frame_index": 0,
                    "end_frame_index": 20,
                },
                {"instruction": "b", "skill": "", "start_frame_index": 20, "end_frame_index": 90},
                {"instruction": "c", "skill": "", "start_frame_index": 95, "end_frame_index": 99},
            ]
        }
    }
    frame = {"episode": 4, "start": 10, "end": 60}
    assert agibot.clip_segments(info, frame) == [
        {"skill": "Pick", "instruction": "a", "start": 0, "end": 10},
        {"skill": "", "instruction": "b", "start": 10, "end": 50},
    ]


# ------------------------------------------------------------------ galaxea


def test_galaxea_english_side_and_runs() -> None:
    assert galaxea.english(" 右手拿起@Pick up the dice.") == "Pick up the dice."
    assert galaxea.english(" organize toys ") == "organize toys"
    assert galaxea.runs(["a", "a", "b", "a"]) == [
        {"instruction": "a", "start": 0, "end": 2},
        {"instruction": "b", "start": 2, "end": 3},
        {"instruction": "a", "start": 3, "end": 4},
    ]


def test_galaxea_schema() -> None:
    features = {n: {"shape": [d]} for n, d in (*galaxea.STATE_LAYOUT, *galaxea.ACTION_LAYOUT)}
    assert galaxea.schema_ok({"features": features})
    features["observation.state.left_arm"] = {"shape": [7]}
    assert not galaxea.schema_ok({"features": features})


def _galaxea_tarball(src: Path, rates: list[float]) -> None:
    """An extracted tarball: one qualified episode of 4 rows per entry of ``rates``, whose
    video is stamped (through ``video_fps``) at that rate."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    layout = (*galaxea.STATE_LAYOUT, *galaxea.ACTION_LAYOUT)
    features = {n: {"dtype": "float32", "shape": [d]} for n, d in layout}
    features["observation.images.head"] = {"dtype": "video", "shape": [8, 8, 3]}
    info = {
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/ep_{episode_index}.mp4",
        "features": features,
    }
    (src / "meta").mkdir(parents=True)
    (src / "data" / "chunk-000").mkdir(parents=True)
    (src / "meta" / "info.json").write_text(json.dumps(info))
    tasks = [{"task_index": 0, "task": "boil the water"}, {"task_index": 1, "task": "qualified"}]
    (src / "meta" / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    eps = [{"episode_index": i, "length": 4} for i in range(len(rates))]
    (src / "meta" / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in eps))
    for i in range(len(rates)):
        cols = {n: pa.array([[0.0] * d] * 4, pa.list_(pa.float32(), d)) for n, d in layout}
        cols |= {c: pa.array([0] * 4) for c in ("task_index", "coarse_task_index")}
        cols |= {c: pa.array([1] * 4) for c in ("quality_index", "coarse_quality_index")}
        pq.write_table(pa.table(cols), src / "data" / "chunk-000" / f"episode_{i:06d}.parquet")


def test_galaxea_leaves_out_episodes_at_another_video_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rates = [15.0, 62.0, 15.0]
    _galaxea_tarball(tmp_path, rates)
    monkeypatch.setattr(galaxea, "video_fps", lambda p: rates[int(p.stem.split("_")[1])])
    off_rate: list[int] = []
    kept = [ep.index for ep, _ in galaxea.episodes_of(tmp_path, text_rules("galaxea"), off_rate)]
    assert (kept, off_rate) == ([0, 2], [1])


def test_galaxea_pin_leaves_out_the_62fps_session() -> None:
    (skipped,) = galaxea.FPS_SKIPPED.values()
    assert len(skipped) == 17
    pin = galaxea.PINS["galaxea"]
    assert (pin.episodes, pin.frames) == (16569 - 17, 18761497 - 89471)


# ------------------------------------------------------------------ rdt


def _split_archive(tmp_path: Path, members: dict[str, bytes], n_parts: int) -> list[Path]:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=0) as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    raw = buf.getvalue()
    size = -(-len(raw) // n_parts)
    parts = []
    for i in range(n_parts):
        part = tmp_path / f"a.tar.gz.{i:02d}"
        part.write_bytes(raw[i * size : (i + 1) * size])
        parts.append(part)
    return parts


def test_rdt_untar_whole_and_first_part(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    sizes = {"a": 100, "b": 4000, "c": 4000}
    members = {
        f"rdt_data/{t}/episode_{i}.hdf5": rng.integers(0, 255, n, dtype=np.uint8).tobytes()
        for t, n in sizes.items()
        for i in range(2)
    }
    parts = _split_archive(tmp_path, members, 3)

    rdt.untar(parts, tmp_path / "all", ("a", "b", "c"), whole=True)
    for name, data in members.items():
        assert (tmp_path / "all" / name).read_bytes() == data

    rdt.untar(parts[:1], tmp_path / "first", ("a",), whole=False)
    assert sorted(p.name for p in (tmp_path / "first" / "rdt_data" / "a").iterdir()) == [
        "episode_0.hdf5",
        "episode_1.hdf5",
    ]
    with pytest.raises(ValueError, match="not whole"):
        rdt.untar(parts[:1], tmp_path / "late", ("c",), whole=False)


def test_rdt_task_text(tmp_path: Path) -> None:
    task = tmp_path / "close_glasses_box"
    task.mkdir()
    assert rdt.task_text(task) == "Close glasses box."
    (task / rdt.INSTRUCTION).write_text(json.dumps({"instruction": " Close the glasses case. "}))
    assert rdt.task_text(task) == "Close the glasses case."
