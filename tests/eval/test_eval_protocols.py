"""launchers/eval_protocols.py: the released eval protocol of each benchmark, built against
synthetic simulator trees (no simulator or dataset on disk)."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import eval_protocols as P
import fire
import pytest

from reki.eval.protocols import CLEAN2RAND_KEYS, EVAL_BENCH_DIRS
from reki.eval.robocasa365_protocol import SUITE_OF, TARGET50

pytestmark = pytest.mark.cpu

REPO = Path(__file__).resolve().parents[2]
CLEAN_YML = (
    "head_camera_type: Large_D435\nwrist_camera_type: Large_D435\nrandom_background: false\n"
    "clean_background_rate: 1\nrandom_light: false\ncrazy_random_light_rate: 0\n"
    "cluttered_table: false\nrandom_table_height: 0\n"
)


# The task list of an exported checkpoint (RoboTwin reads it).
SPEC_TASKS = [f"task{i}" for i in range(8)]


@pytest.fixture
def sims(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    sim = tmp_path / "sim"
    for d in (
        "libero/assets/scenes",
        "libero_plus/LIBERO-plus/libero/libero/benchmark",
        "libero_plus/LIBERO-plus/libero/libero/assets/scenes",
        "robotwin/RoboTwin/task_config",
        "robotwin/mamba/envs/robotwin",
        "robocasa365/mamba/envs/robocasa365",
        "bin",
    ):
        (sim / d).mkdir(parents=True)
    (sim / "libero_plus/LIBERO-plus/libero/libero/benchmark/task_classification.json").touch()
    (sim / "bin/micromamba").touch()
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "ckpt/reki_inference.json").write_text(json.dumps({"tasks": SPEC_TASKS}))
    for name in ("demo_clean", "demo_randomized"):
        (sim / f"robotwin/RoboTwin/task_config/{name}.yml").write_text(CLEAN_YML)
    for lib in ("osmesa.so", "jpeg.so"):
        (tmp_path / lib).touch()
    env = {"REKI_SIM_ROOT": "sim", "OSMESA_PRELOAD": "osmesa.so", "LIBJPEG_PRELOAD": "jpeg.so"}
    for k, v in env.items():
        monkeypatch.setenv(k, str(tmp_path / v))
    return tmp_path


def _flags(job: P.EvalJob) -> dict[str, str]:
    toks = shlex.split(job.command())
    i = next(i for i, t in enumerate(toks) if t.endswith("/scripts/eval/rollout.py"))
    return dict(t[2:].split("=", 1) for t in toks[i + 1 :])


@pytest.mark.parametrize(
    "task, n_jobs, episodes",
    [
        ("libero", 4, "50"),  # one job per suite
        ("libero_plus", 28, "1"),
        ("robotwin", 4, "20"),  # 8 tasks in packs of 6, per difficulty
        ("robotwin_clean2rand", 8, "20"),
        ("robocasa365", 9, "50"),  # 18/16/16 tasks per suite, packs of 6
    ],
)
def test_protocol_defaults(sims: Path, task: str, n_jobs: int, episodes: str) -> None:
    jobs = P.eval_jobs(task, str(sims / "ckpt"), "/out", use_ema=True)
    assert len(jobs) == n_jobs
    assert len({j.prefix for j in jobs}) == 1
    for job in jobs:
        f = _flags(job)
        assert (f["n_episodes"], f["use_ema"]) == (episodes, "True")
        assert "n_action_steps" not in f  # rollout.py picks the checkpoint's cadence
        assert "ema_name" not in f and f["save_videos"] == "False"
        assert any(f["log_dir"] == f"/out/{d}" for d in EVAL_BENCH_DIRS[task])


def test_robotwin_server_configs_and_clean2rand_files(sims: Path) -> None:
    jobs = P.eval_jobs("robotwin", str(sims / "ckpt"), "/out", use_ema=False)
    easy, hard = (next(j for j in jobs if f"/{d}/" in j.name) for d in ("easy", "hard"))
    assert "--task-config demo_clean " in _flags(easy)["sim_server_cmd"]
    assert "--task-config demo_randomized " in _flags(hard)["sim_server_cmd"]

    P.eval_jobs("robotwin_clean2rand", str(sims / "ckpt"), "/out", use_ema=False)
    cfg = sims / "sim/robotwin/RoboTwin/task_config"
    clean = (cfg / "demo_clean.yml").read_text().splitlines()
    for axis, keys in CLEAN2RAND_KEYS.items():
        derived = (cfg / f"demo_{axis}.yml").read_text().splitlines()
        changed = {a.split(":")[0] for a, b in zip(clean, derived) if a != b}
        assert changed == set(keys), axis


def test_robotwin_refuses_low_res_cameras(sims: Path) -> None:
    yml = sims / "sim/robotwin/RoboTwin/task_config/demo_clean.yml"
    yml.write_text(CLEAN_YML.replace("head_camera_type: Large_D435", "head_camera_type: D435"))
    with pytest.raises(SystemExit, match="Large_D435"):
        P.eval_jobs("robotwin", str(sims / "ckpt"), "/out", use_ema=False)


def test_robotwin_remote_ports(sims: Path) -> None:
    (job,) = P.eval_jobs(
        "robotwin", str(sims / "ckpt"), "/out", use_ema=False, difficulties="easy", sim_port="10-12"
    )
    f = _flags(job)
    assert (f["sim_host"], f["sim_port"], f["num_workers"]) == ("127.0.0.1", "10,11,12", "3")
    assert "sim_server_cmd" not in f
    # One job per num_workers ports, the tasks dealt round-robin.
    jobs = P.eval_jobs(
        "robotwin",
        str(sims / "ckpt"),
        "/out",
        use_ema=False,
        difficulties="easy",
        sim_port="10-13",
        num_workers=2,
    )
    flags = [_flags(j) for j in jobs]
    assert [(f["sim_port"], f["num_workers"]) for f in flags] == [("10,11", "2"), ("12,13", "2")]
    tasks = [f["task"].split(",") for f in flags]
    assert sorted(tasks[0] + tasks[1]) == sorted(_flags(job)["task"].split(","))
    assert abs(len(tasks[0]) - len(tasks[1])) <= 1
    with pytest.raises(SystemExit, match="do not divide"):
        P.eval_jobs(
            "robotwin",
            str(sims / "ckpt"),
            "/out",
            use_ema=False,
            difficulties="easy",
            sim_port="10-14",
            num_workers=2,
        )
    with pytest.raises(SystemExit, match="one task config"):
        P.eval_jobs("robotwin", str(sims / "ckpt"), "/out", use_ema=False, sim_port=10)
    with pytest.raises(SystemExit, match="robotwin tasks"):
        P.eval_jobs("robocasa365", str(sims / "ckpt"), "/out", use_ema=False, sim_port=10)


def test_robocasa365_server_runs_without_numba_jit(sims: Path) -> None:
    job = P.eval_jobs("robocasa365", str(sims / "ckpt"), "/out", use_ema=False)[0]
    server = _flags(job)["sim_server_cmd"]
    assert "NUMBA_DISABLE_JIT=1" in server and "LD_PRELOAD" in server
    assert "robocasa365_sim_server.py --port 0" in server and " && " not in server


@pytest.mark.parametrize("spec", ["14-10", "", "10,", "a"])
def test_bad_sim_port(spec: str) -> None:
    with pytest.raises(SystemExit, match="expected ports"):
        P.parse_ports(spec)


@pytest.mark.parametrize("task", ["libero", "libero_plus", "robotwin", "robocasa365"])
def test_missing_install_says_how(sims: Path, task: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REKI_SIM_ROOT", str(sims / "empty"))
    with pytest.raises(SystemExit, match=f"bash scripts/eval/install.sh {task} "):
        P.eval_jobs(task, str(sims / "ckpt"), "/out", use_ema=False)


def test_remote_robotwin_needs_no_install(sims: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REKI_SIM_ROOT")
    jobs = P.eval_jobs(
        "robotwin", str(sims / "ckpt"), "/out", use_ema=False, difficulties="easy", sim_port=10
    )
    assert _flags(jobs[0])["sim_port"] == "10"


@pytest.mark.parametrize("tasks", ["adjust_bottle", "adjust_bottle,beat_block_hammer"])
def test_cli_task_list(sims: Path, tasks: str, capsys: pytest.CaptureFixture[str]) -> None:
    fire.Fire(P.main, ["robotwin", str(sims / "ckpt"), "/out", "--tasks", tasks])
    assert f"--task={tasks} " in capsys.readouterr().out


def test_tuple_units_from_grid_verbs(sims: Path) -> None:
    """Fire hands the grid verbs ``--tasks a,b`` as a tuple; eval_jobs takes it as is."""
    jobs = P.eval_jobs(
        "robotwin",
        str(sims / "ckpt"),
        "/out",
        use_ema=False,
        tasks=("beat_block_hammer", "stack_bowls_two"),
        difficulties=("easy", "hard"),
    )
    assert [j.command() for j in jobs] == [
        j.command()
        for j in P.eval_jobs(
            "robotwin",
            str(sims / "ckpt"),
            "/out",
            use_ema=False,
            tasks="beat_block_hammer,stack_bowls_two",
            difficulties="easy,hard",
        )
    ]
    assert [_flags(j)["task"] for j in jobs] == ["beat_block_hammer,stack_bowls_two"] * 2
    (libero,) = P.eval_jobs(
        "libero", str(sims / "ckpt"), "/out", use_ema=False, tasks=("libero_goal",)
    )
    assert "libero_goal" in libero.command()


def test_packs_cover_each_task_once(sims: Path) -> None:
    """One worker per task; each task in one job; RoboCasa365 packs stay within a suite."""
    jobs = P.eval_jobs("robotwin", str(sims / "ckpt"), "/out", use_ema=False, difficulties="easy")
    packs = [_flags(j)["task"].split(",") for j in jobs]
    assert [len(p) for p in packs] == [6, 2]
    assert sum(packs, []) == SPEC_TASKS
    assert [_flags(j)["num_workers"] for j in jobs] == ["6", "2"]
    jobs = P.eval_jobs("robocasa365", str(sims / "ckpt"), "/out", use_ema=False)
    packs = [_flags(j)["task"].split(",") for j in jobs]
    assert sorted(sum(packs, [])) == sorted(TARGET50)
    assert all(len({SUITE_OF[t] for t in p}) == 1 for p in packs)


def test_cli_short_form(sims: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fire.Fire(P.main, ["robotwin_easy", str(sims / "ckpt"), "/out"])
    assert "--log_dir=/out/robotwin_easy" in capsys.readouterr().out


def test_hf_id_reads_the_spec(sims: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A Hugging Face id (no such local dir) takes its task list from the repo's spec."""
    import huggingface_hub

    spec = sims / "ckpt/reki_inference.json"
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, name: str(spec))
    jobs = P.eval_jobs("robotwin", "kyutai/some-model", "/out", use_ema=False, difficulties="easy")
    assert sum((_flags(j)["task"].split(",") for j in jobs), []) == SPEC_TASKS


def _server_cmd(sims: Path, task_config: str) -> str:
    rt = sims / "sim/robotwin"
    return (
        f"PYTHONNOUSERSITE=1 PYTHONPATH= NVIDIA_DRIVER_CAPABILITIES=all "
        f"CUDA_HOME={rt}/mamba/envs/robotwin XDG_CACHE_HOME={{portfile}}.mamba_cache "
        f"{sims}/sim/bin/micromamba run -r {rt}/mamba -n robotwin python -s "
        f"{REPO}/scripts/eval/robotwin_sim_server.py --repo {rt}/RoboTwin "
        f"--task-config {task_config} "
        f"--port 0 --port-file {{portfile}} > {{portfile}}.log 2>&1"
    )


def test_robotwin_job_args(sims: Path) -> None:
    ckpt = str(sims / "ckpt")
    common = [("ckpt_dir", ckpt), ("n_episodes", 20), ("use_ema", True)]
    common += [("save_videos", False)]
    head = [("env_type", "robotwin")]
    local = P.eval_jobs("robotwin", ckpt, "/out", use_ema=True, difficulties="easy")[0]
    assert local.name == "robotwin/easy/pack00"
    assert list(local.args.items()) == [
        *head,
        ("task", ",".join(SPEC_TASKS[:6])),
        *common,
        ("num_workers", 6),
        ("sim_server_cmd", _server_cmd(sims, "demo_clean")),
        ("log_dir", Path("/out/robotwin_easy")),
    ]
    (remote,) = P.eval_jobs(
        "robotwin", ckpt, "/out", use_ema=True, difficulties="easy", sim_port="10-12"
    )
    assert remote.name == "robotwin/easy"
    assert list(remote.args.items()) == [
        *head,
        ("task", ",".join(SPEC_TASKS)),
        *common,
        ("num_workers", 3),
        ("sim_port", "10,11,12"),
        ("log_dir", Path("/out/robotwin_easy")),
    ]
    (c2r,) = P.eval_jobs(
        "robotwin_clean2rand", ckpt, "/out", use_ema=False, difficulties="light", tasks="a"
    )
    assert _flags(c2r)["log_dir"] == "/out/robotwin_light"


@pytest.mark.parametrize("flags", [{"resume": "submit"}, {"ema": "maybe"}, {"resume": 2}])
def test_check_bools_rejects_non_bool(flags: dict[str, object]) -> None:
    with pytest.raises(SystemExit, match="write --"):
        P.check_bools(**flags)


def test_check_bools_coerces():
    # fire passes --ema=false as "false" and --resume=0 as 0.
    assert P.check_bools(ema=True, resume="false", save_videos=0, sim_relay="TRUE") == [
        True,
        False,
        False,
        True,
    ]
