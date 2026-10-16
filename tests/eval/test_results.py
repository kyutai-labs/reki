import json
from pathlib import Path

import pytest

import reki.eval.protocols as P
import reki.eval.results as tab
from reki.eval.robocasa365_protocol import TARGET_SUITES
from reki.inference.spec import INFERENCE_SPEC_FILE

pytestmark = pytest.mark.cpu
TIDS = [f"{t:02d}" for t in range(10)]


def _leaf(
    root: Path, rel: str, n_success: int, n_episodes: int = 2, ckpt: Path | None = None
) -> None:
    d = root / rel
    d.mkdir(parents=True)
    run: dict[str, int | str | list] = {
        "n_episodes": n_episodes,
        "n_success": n_success,
        "target_episodes": 2,
        "episodes": [],
    }
    if ckpt is not None:
        run["ckpt_dir"] = str(ckpt)
    (d / "results.json").write_text(json.dumps(run))


def _exported_ckpt(root: Path, tasks: list[str]) -> Path:
    ckpt = root / "ckpt"
    ckpt.mkdir()
    (ckpt / INFERENCE_SPEC_FILE).write_text(json.dumps({"tasks": tasks}))
    return ckpt


def test_missing_libero_suite_marks_partial(tmp_path: Path) -> None:
    _leaf(tmp_path, "libero/libero_object/00", 2)
    s = tab.summarize(tab.collect(tmp_path))["libero"]
    assert s["partial"]
    assert s["missing_groups"] == sorted(x for x in P.LIBERO_SUITES if x != "libero_object")


def test_full_libero_is_not_partial(tmp_path: Path) -> None:
    for suite in P.LIBERO_SUITES:
        for tid in TIDS:
            _leaf(tmp_path, f"libero/{suite}/{tid}", 1)
    s = tab.summarize(tab.collect(tmp_path))["libero"]
    assert not s["partial"] and s["missing_groups"] == []
    assert s["macro"] == 0.5


def test_missing_libero_plus_axis_marks_partial(tmp_path: Path) -> None:
    for suite in P.LIBERO_SUITES:
        for axis in P.LIBERO_PLUS_AXES[1:]:
            _leaf(tmp_path, f"libero_plus/{P.libero_plus_group(suite, axis)}/0", 2)
    s = tab.summarize(tab.collect(tmp_path))["libero_plus"]
    assert s["partial"]
    assert len(s["missing_groups"]) == len(P.LIBERO_SUITES)


def test_short_unit_marks_partial(tmp_path: Path) -> None:
    for suite in P.LIBERO_SUITES:
        for tid in TIDS:
            _leaf(tmp_path, f"libero/{suite}/{tid}", 1, n_episodes=1 if tid == "09" else 2)
    s = tab.summarize(tab.collect(tmp_path))["libero"]
    assert s["partial"] and all(g["missing_units"] == [] for g in s["groups"].values())


def test_missing_libero_task_id_marks_partial(tmp_path: Path) -> None:
    for suite in P.LIBERO_SUITES:
        for tid in TIDS:
            if (suite, tid) != ("libero_goal", "03"):
                _leaf(tmp_path, f"libero/{suite}/{tid}", 1)
    s = tab.summarize(tab.collect(tmp_path))["libero"]
    assert s["partial"] and s["groups"]["libero_goal"]["missing_units"] == ["03"]
    assert not s["groups"]["libero_object"]["partial"]


@pytest.mark.parametrize("bench", ["robotwin_easy", "robotwin_background"])
def test_missing_socket_task_marks_partial(tmp_path: Path, bench: str) -> None:
    ckpt = _exported_ckpt(tmp_path, ["task_a", "task_b", "task_c"])
    for task in ("task_a", "task_c"):
        _leaf(tmp_path, f"out/{bench}/{task}", 1, ckpt=ckpt)
    s = tab.summarize(tab.collect(tmp_path / "out"))[bench]
    assert s["partial"] and s["groups"]["all"]["missing_units"] == ["task_b"]
    _leaf(tmp_path, f"out/{bench}/task_b", 1, ckpt=ckpt)
    assert not tab.summarize(tab.collect(tmp_path / "out"))[bench]["partial"]


def test_missing_robocasa365_task_marks_partial(tmp_path: Path) -> None:
    for suite, tasks in TARGET_SUITES.items():
        for task in tasks[1:]:
            _leaf(tmp_path, f"robocasa365/{task}", 2)
    s = tab.summarize(tab.collect(tmp_path))["robocasa365"]
    assert s["partial"] and s["missing_groups"] == []
    for suite, tasks in TARGET_SUITES.items():
        assert s["groups"][suite]["missing_units"] == [tasks[0]]


def test_libero_plus_score_is_pooled_over_episodes(tmp_path: Path) -> None:
    suite = P.LIBERO_SUITES[0]
    a, b = P.LIBERO_PLUS_AXES[0], P.LIBERO_PLUS_AXES[1]
    _leaf(tmp_path, f"libero_plus/{P.libero_plus_group(suite, a)}/0", 1, n_episodes=2)
    for i in range(4):
        _leaf(tmp_path, f"libero_plus/{P.libero_plus_group(suite, b)}/{i}", 2, n_episodes=2)
    s = tab.summarize(tab.collect(tmp_path))["libero_plus"]
    assert s["macro"] == pytest.approx(9 / 10)
    assert s["per_axis"][a] == pytest.approx(0.5)
    assert s["per_axis"][b] == pytest.approx(1.0)


def test_clean2rand_mean_of_six_scenes(tmp_path: Path) -> None:
    ckpt = _exported_ckpt(tmp_path, ["task_a", "task_b"])
    scene_dirs = P.CLEAN2RAND_SCENE_DIRS
    assert scene_dirs[0] == "robotwin_easy" and scene_dirs[-1] == "robotwin_hard"
    for i, bench in enumerate(scene_dirs[:-1]):
        for task in ("task_a", "task_b"):
            _leaf(tmp_path, f"out/{bench}/{task}", i % 3, ckpt=ckpt)
    cells = tab.clean2rand_cells(tmp_path / "out", scene_dirs)
    assert cells["avg"].startswith("[dim]")  # Hard has no results yet
    assert cells["trials"] == "[dim]20/24[/dim]"
    for task in ("task_a", "task_b"):
        _leaf(tmp_path, f"out/{scene_dirs[-1]}/{task}", 2, ckpt=ckpt)
    cells = tab.clean2rand_cells(tmp_path / "out", scene_dirs)
    # Scene task means 0, 50, 100, 0, 50 (i % 3 of 2 episodes) and 100 (Hard).
    assert cells == {"trials": "24/24", "avg": "[bold]50.0[/bold]"}


def test_result_cells_grey_partial_and_count_trials(tmp_path: Path) -> None:
    for tid in TIDS:
        _leaf(tmp_path, f"libero/libero_object/{tid}", 1)
    cells = tab.result_cells(tmp_path, ("libero",))
    assert cells["libero"].startswith("[dim]")
    assert cells["steps"] == "-"  # the leaves list no episodes
    n = len(TIDS) * 2  # the episodes _leaf writes per suite
    assert cells["trials"] == f"[dim]{n}/{n * len(P.LIBERO_SUITES)}[/dim]"


def test_result_cells_mean_steps_of_successes(tmp_path: Path) -> None:
    d = tmp_path / "libero/libero_goal/00"
    d.mkdir(parents=True)
    episodes = [
        {"steps": 100, "success": True},
        {"steps": 200, "success": True},
        {"steps": 300, "success": False},
    ]
    run = {"n_episodes": 3, "n_success": 2, "target_episodes": 3, "episodes": episodes}
    (d / "results.json").write_text(json.dumps(run))
    assert tab.result_cells(tmp_path, ("libero",))["steps"] == "[dim]150[/dim]"


def test_libero_plus_trials_count_every_task(tmp_path: Path) -> None:
    """The expected total counts the 10,030 tasks before their results exist."""
    _leaf(tmp_path, "libero_plus/libero_plus_goal/camera_viewpoints/0", 1)
    cells = tab.result_cells(tmp_path, ("libero_plus",))
    assert cells["trials"] == f"[dim]2/{10030 * 2}[/dim]"  # _leaf: 2 episodes per unit
    assert "cam" in cells
