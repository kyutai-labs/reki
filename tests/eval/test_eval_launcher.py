import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from reki.eval.protocols import EVAL_BENCH_DIRS, TASK_BENCHMARK

eval_launcher = importlib.import_module("eval")

pytestmark = pytest.mark.cpu


def test_every_task_maps_to_one_benchmark() -> None:
    assert set(TASK_BENCHMARK) == set(EVAL_BENCH_DIRS)
    for name, model in eval_launcher.MODELS.items():
        benchmarks = {TASK_BENCHMARK[t] for t in model["tasks"]}
        assert len(benchmarks) == 1, name


def test_resolve_needs_an_export(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="reki_inference.json"):
        eval_launcher.resolve(str(tmp_path))
    (tmp_path / "reki_inference.json").write_text("{}")
    assert eval_launcher.resolve(str(tmp_path)) == tmp_path


def _grid(tmp_path: Path, ckpt: Path) -> tuple[Any, dict[str, dict[str, Any]]]:
    models = {"m": {"ckpt": str(ckpt), "tasks": ["robocasa365"]}}
    grid = eval_launcher.ModelGrid(
        base_name="models",
        base_cmd="false",
        log_dir=str(tmp_path / "runs"),
        use_hash_in_dirnames=True,
        expand_groups_in_name=False,
        git_sync=False,
        setup={"m": {"ckpt": str(ckpt), "tasks": "robocasa365"}},
    )
    return grid, models


def test_preview_verbs_neither_download_nor_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    grid, models = _grid(tmp_path, Path("org/not-downloaded"))
    monkeypatch.setattr(eval_launcher, "MODELS", models)
    monkeypatch.setattr(eval_launcher, "resolve", lambda ckpt: pytest.fail("resolved"))
    (exp,) = grid
    assert grid._eval_targets(exp, None, False) == [("m", "org/not-downloaded", exp.log_dir)]
    assert not (Path(exp.log_dir) / eval_launcher.SOURCE_FILE).exists()


def test_submit_records_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ckpt = tmp_path / "export"
    ckpt.mkdir()
    (ckpt / "reki_inference.json").write_text(json.dumps({"benchmark": "robocasa365"}))
    grid, models = _grid(tmp_path, ckpt)
    monkeypatch.setattr(eval_launcher, "MODELS", models)
    (exp,) = grid
    grid._submitting = True
    assert grid._eval_targets(exp, None, False) == [("m", str(ckpt), exp.log_dir)]
    source = Path(exp.log_dir) / eval_launcher.SOURCE_FILE
    assert json.loads(source.read_text())["path"] == str(ckpt)
    grid._submitting = False
    assert grid._eval_targets(exp, None, False)[0][1] == str(ckpt)
    assert grid._result_dirs(exp, None, False) == [("local", Path(exp.log_dir))]


def test_results_without_source_fail_loud(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    grid, models = _grid(tmp_path, Path("org/repo"))
    monkeypatch.setattr(eval_launcher, "MODELS", models)
    (exp,) = grid
    assert grid._result_dirs(exp, None, False) == []
    leaf = Path(exp.log_dir) / "robocasa365/task"
    leaf.mkdir(parents=True)
    (leaf / "results.json").write_text("{}")
    with pytest.raises(SystemExit, match="source.json"):
        grid._result_dirs(exp, None, False)


def _rc365_leaf(out: Path, task: str, n_episodes: int, n_success: int) -> None:
    leaf = out / "robocasa365" / task
    leaf.mkdir(parents=True)
    leaf.joinpath("results.json").write_text(
        json.dumps(
            {
                "n_episodes": n_episodes,
                "n_success": n_success,
                "target_episodes": 50,
                "episodes": [],
            }
        )
    )


def test_task_rates_lists_every_task(tmp_path: Path) -> None:
    common = importlib.import_module("common")
    from reki.eval.robocasa365_protocol import TARGET50

    done, short = list(TARGET50)[:2]
    _rc365_leaf(tmp_path, done, 50, 40)
    _rc365_leaf(tmp_path, short, 10, 5)
    rates = common._task_rates(tmp_path, "robocasa365")
    assert set(rates) == set(TARGET50)
    assert rates[done] == "80.0" and "50.0" in rates[short]
    assert sum(v == "-" for v in rates.values()) == len(TARGET50) - 2


def test_eval_skips_runs_without_the_step(tmp_path: Path) -> None:
    common = importlib.import_module("common")
    grid = common.RekiGrid(
        base_name="sft",
        base_cmd="false",
        log_dir=str(tmp_path / "runs"),
        use_hash_in_dirnames=True,
        git_sync=False,
        setup={"a": {"steps": 1}, "b": {"steps": 2}},
    )
    a, b = grid
    for exp, step in ((a, 5000), (b, 10000)):
        (Path(exp.log_dir) / "train" / f"checkpoint-{step}").mkdir(parents=True)
    assert grid._eval_targets(a, 10000, ema=False) == []
    ((_, ckpt, out),) = grid._eval_targets(b, 10000, ema=False)
    assert ckpt.endswith("checkpoint-10000") and out.endswith("checkpoint-10000/raw")


def _sft_grid(tmp_path: Path) -> Any:
    common = importlib.import_module("common")
    return common.RekiGrid(
        base_name="sft",
        base_cmd="false",
        log_dir=str(tmp_path / "runs"),
        use_hash_in_dirnames=True,
        git_sync=False,
        setup={"a": {"steps": 1}},
    )


def _robotwin_leaf(out: Path, bench: str, task: str, n_success: int, ckpt: Path) -> None:
    leaf = out / bench / task
    leaf.mkdir(parents=True)
    run = {
        "n_episodes": 2,
        "n_success": n_success,
        "target_episodes": 2,
        "ckpt_dir": str(ckpt),
        "episodes": [],
    }
    leaf.joinpath("results.json").write_text(json.dumps(run))


def test_results_rows_per_difficulty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("COLUMNS", "200")
    grid = _sft_grid(tmp_path)
    (exp,) = grid
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "reki_inference.json").write_text(json.dumps({"tasks": ["t0", "t1"]}))
    out = Path(exp.log_dir) / "eval/checkpoint-1000/ema"
    for bench in ("robotwin_easy", "robotwin_hard"):
        for task in ("t0", "t1"):
            _robotwin_leaf(out, bench, task, 1, ckpt)

    def table(task: str) -> list[str]:
        grid.results(task=task)
        return [ln.strip("│ ") for ln in capsys.readouterr().out.splitlines()]

    rows = table("robotwin")
    assert any("easy" in ln and "50.0" in ln and "4/4" in ln for ln in rows)
    assert any(" hard " in ln for ln in rows)
    short = table("robotwin_easy")
    assert any(" easy " in ln for ln in short) and not any(" hard " in ln for ln in short)


def test_eval_checks_the_servers_of_its_scene(tmp_path: Path) -> None:
    grid = _sft_grid(tmp_path)
    (exp,) = grid
    ckpt = Path(exp.log_dir) / "train/checkpoint-1000"
    ckpt.mkdir(parents=True)
    (ckpt / "reki_inference.json").write_text(json.dumps({"tasks": ["t0", "t1"]}))
    jobs = grid.eval(task="robotwin_background", ema=False, sim_port="10-11", sim_relay=True)
    assert jobs.relay_ports == (10, 11)
    assert jobs.relay_log_dir == Path(exp.log_dir) / "eval/checkpoint-1000/raw"
    assert jobs.sim_check == ("demo_background", "t0")


def test_eval_requires_opt_in_for_sim_relay(tmp_path: Path) -> None:
    grid = _sft_grid(tmp_path)
    (exp,) = grid
    ckpt = Path(exp.log_dir) / "train/checkpoint-1000"
    ckpt.mkdir(parents=True)
    (ckpt / "reki_inference.json").write_text(json.dumps({"tasks": ["t0"]}))
    with pytest.raises(SystemExit, match="--sim_relay=True"):
        grid.eval(task="robotwin_easy", ema=False, sim_port=10)


def _fake_sim_server(task_config: str) -> int:
    """A loopback server that answers one make per connection, then a close."""
    import socket
    import threading

    from reki.eval import sim_protocol

    srv = socket.create_server(("127.0.0.1", 0))

    def serve() -> None:
        while True:
            conn, _ = srv.accept()
            with conn:
                sim_protocol.recv(conn)
                code = {"task_config": task_config}
                sim_protocol.send(conn, {"code": code})
                sim_protocol.recv(conn)

    threading.Thread(target=serve, daemon=True).start()
    return srv.getsockname()[1]


def test_check_sim_servers() -> None:
    import socket

    common = importlib.import_module("common")
    good, wrong = _fake_sim_server("demo_light"), _fake_sim_server("demo_clean")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dead = s.getsockname()[1]
    common.check_sim_servers((good,), "demo_light", "t0")
    with pytest.raises(SystemExit) as e:
        common.check_sim_servers((good, wrong, dead), "demo_light", "t0")
    msg = str(e.value)
    assert f"{wrong}: serves demo_clean" in msg
    assert f"{dead}: no sim server" in msg
    assert f"{good}:" not in msg
