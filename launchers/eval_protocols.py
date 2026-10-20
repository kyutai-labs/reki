"""Released eval protocols: the ``scripts/eval/rollout.py`` command(s) for one checkpoint on one
benchmark. ``launchers/common.py`` submits them as 1-GPU Slurm jobs; without Slurm, print them:

    pixi run python launchers/eval_protocols.py <task> <ckpt_dir> <out_dir>

Results land under ``<out_dir>/<bench dir>/`` (``reki.eval.protocols.EVAL_BENCH_DIRS``); tabulate
them with ``scripts/eval/results_table.py <out_dir>``. Socket benchmarks (RoboTwin, RoboCasa365)
start their own sim server per worker (``--sim_server_cmd``). For RoboTwin, ``--sim_port`` connects
to servers that already run (``--sim_host``, default 127.0.0.1): one job per ``num_workers``
ports, one worker per port.

Jobs: LIBERO one per suite, LIBERO-Plus one per suite and axis, RoboTwin and RoboCasa365 one per
``num_workers`` tasks (one worker per task; RoboCasa365 packs stay within one suite).

The simulators are under ``$REKI_SIM_ROOT`` (default ``<repo>/storage/sim``), one directory per
benchmark: ``bash scripts/eval/install.sh <benchmark>`` installs one (scripts/eval/SETUP.md).
RoboTwin with ``--sim_port`` needs no install on the eval machine.

An exported checkpoint (``scripts/export_checkpoint.py --benchmark``) carries its
normalization stats; a raw training checkpoint reads them from ``REKI_DATASETS_ROOT``.
"""

from __future__ import annotations

import os
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path

from reki.eval.protocols import (
    CLEAN2RAND_KEYS,
    EVAL_BENCH_DIRS,
    LIBERO_PLUS_AXES,
    LIBERO_SUITES,
    N_EPISODES,
    ROBOTWIN_TASK_CONFIG,
    TASK_BENCHMARK,
    libero_plus_category,
    libero_plus_group,
    socket_bench_tasks,
)
from reki.eval.robocasa365_protocol import SUITE_OF
from reki.eval.robocasa365_protocol import TARGET_SUITES as ROBOCASA365_SUITES
from reki.paths import sim_root
from reki.utils.cli import fire_bool

REPO = Path(__file__).resolve().parent.parent
# One RoboTwin scene as a task of its own, e.g. for servers that run one task config.
TASK_SHORT_FORMS = {
    "robotwin_easy": ("robotwin", "easy"),
    "robotwin_hard": ("robotwin", "hard"),
    **{f"robotwin_{a}": ("robotwin_clean2rand", a) for a in CLEAN2RAND_KEYS},
}


def expand_short_task[T: str | None, D](task: T, difficulties: D) -> tuple[T | str, D | str]:
    """(task, difficulties) with a TASK_SHORT_FORMS task replaced by what it stands for."""
    if task not in TASK_SHORT_FORMS:
        return task, difficulties
    if difficulties:
        raise SystemExit(f"--task {task} already sets --difficulties")
    return TASK_SHORT_FORMS[task]


EVAL_ROLLOUT = REPO / "scripts/eval/rollout.py"


def _sim(benchmark: str, *must_exist: str) -> Path:
    """``$REKI_SIM_ROOT/<benchmark>``; exit with the install command if a file is missing."""
    root = sim_root() / benchmark
    for rel in must_exist:
        if not (root / rel).exists():
            raise SystemExit(
                f"{root / rel} missing. Install {benchmark}: "
                f"bash scripts/eval/install.sh {benchmark} (scripts/eval/SETUP.md)"
            )
    return root


def _file(path: Path, hint: str) -> Path:
    if not path.exists():
        raise SystemExit(f"{path} missing: {hint}")
    return path


def _micromamba(benchmark: str) -> Path:
    return _file(sim_root() / "bin/micromamba", f"bash scripts/eval/install.sh {benchmark}")


@dataclass
class EvalJob:
    """One rollout.py process: ``prefix`` (shell setup + env + interpreter) + ``args``.
    ``prefix`` is the same for every job of a task, so a Slurm grid shares it. ``scene``: the
    RoboTwin difficulty or Clean2Rand axis whose task config the job's servers run."""

    name: str
    prefix: str
    args: dict[str, object] = field(default_factory=dict)
    scene: str | None = None

    def command(self) -> str:
        flags = " ".join(shlex.quote(f"--{k}={v}") for k, v in self.args.items())
        return f"{self.prefix} {flags}"


def _client_env(extra: dict[str, str]) -> str:
    env = {"TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"}
    for name in ("HF_HOME", "REKI_DATASETS_ROOT"):
        if os.environ.get(name):
            env[name] = os.environ[name]
    env.update(extra)
    return " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())


def _python(pixi_env: str) -> str:
    return (
        f"pixi run --manifest-path {REPO}/pyproject.toml --frozen --no-install -e {pixi_env} python"
    )


def _osmesa() -> str:
    return os.environ.get("OSMESA_PRELOAD") or str(
        _file(
            REPO / ".pixi/envs/sim/lib/libOSMesa.so.8", "pixi install -e sim, or set OSMESA_PRELOAD"
        )
    )


def _common_args(
    ckpt_dir: str,
    n_episodes: int,
    *,
    use_ema: bool,
    resume: bool,
    save_videos: bool,
    history_k: int | None,
    n_action_steps: int | None = None,
) -> dict[str, object]:
    args: dict[str, object] = {
        "ckpt_dir": ckpt_dir,
        "n_episodes": n_episodes,
        "use_ema": use_ema,
    }
    args["save_videos"] = save_videos
    if resume:
        args["resume"] = True
    if history_k is not None:
        args["history_k"] = history_k
    if n_action_steps is not None:
        args["n_action_steps"] = n_action_steps
    return args


def _libero(
    common: dict[str, object], out: Path, num_workers: int, suites: tuple[str, ...]
) -> list[EvalJob]:
    root = _sim("libero", "assets/scenes")
    config = root / "config"
    seed_config = f"bash {REPO}/scripts/eval/libero_config.sh libero {root} --if-missing"
    env = _client_env(
        {"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa", "LIBERO_CONFIG_PATH": str(config)}
    )
    prefix = f"{seed_config} && env -u DISPLAY {env} {_python('sim')} {EVAL_ROLLOUT}"
    return [
        EvalJob(
            f"libero/{suite.removeprefix('libero_')}",
            prefix,
            {
                "env_type": "libero",
                "task": suite,
                **common,
                "num_workers": num_workers,
                "log_dir": out / "libero",
            },
        )
        for suite in suites
    ]


def _libero_plus(
    common: dict[str, object], out: Path, num_workers: int, suites: tuple[str, ...]
) -> list[EvalJob]:
    root = _sim(
        "libero_plus",
        "LIBERO-plus/libero/libero/benchmark/task_classification.json",
        "LIBERO-plus/libero/libero/assets/scenes",
    )
    repo, config = root / "LIBERO-plus", root / "config"
    libjpeg = os.environ.get("LIBJPEG_PRELOAD") or str(
        _file(
            REPO / ".pixi/envs/sim-plus/lib/libjpeg.so.8",
            "pixi install -e sim-plus, or set LIBJPEG_PRELOAD",
        )
    )
    write_config = f"bash {REPO}/scripts/eval/libero_config.sh libero_plus {root}"
    env = _client_env(
        {
            "MUJOCO_GL": "osmesa",
            "PYOPENGL_PLATFORM": "osmesa",
            "LD_PRELOAD": libjpeg,
            "PYTHONPATH": str(repo),
            "LIBERO_CONFIG_PATH": str(config),
        }
    )
    prefix = f"{write_config} && env -u DISPLAY {env} {_python('sim-plus')} {EVAL_ROLLOUT}"
    jobs = []
    for suite in suites:
        for axis in LIBERO_PLUS_AXES:
            args = {
                "env_type": "libero",
                "task": suite,
                "perturbation_axis": libero_plus_category(axis),
                "suite_alias": libero_plus_group(suite, axis),
                **common,
                "num_workers": num_workers,
                "log_dir": out / "libero_plus",
            }
            jobs.append(
                EvalJob(f"libero_plus/{suite.removeprefix('libero_')}/{axis}", prefix, args)
            )
    return jobs


def _derive_clean2rand_config(task_config: Path, axis: str) -> None:
    """Write demo_<axis>.yml from demo_clean.yml, unless it already has that content."""
    dst = task_config / f"demo_{axis}.yml"
    text = (task_config / "demo_clean.yml").read_text()
    for key, value in CLEAN2RAND_KEYS[axis].items():
        text, n = re.subn(rf"(?m)^(\s*{key}:).*$", rf"\g<1> {value}", text)
        if n != 1:
            raise SystemExit(f"{task_config}/demo_clean.yml: key {key!r} found {n} times")
    if dst.exists() and dst.read_text() == text:
        return
    dst.write_text(text)
    print(f"[eval] wrote {dst}", file=sys.stderr)


def parse_ports(spec: str | int | tuple[int, ...]) -> list[int]:
    """``10-14``, ``10,11,12`` or one port (fire parses ``10,11`` as a tuple)."""
    if isinstance(spec, int):
        return [spec]
    ports = []
    for tok in spec if isinstance(spec, tuple) else str(spec).split(","):
        lo, _, hi = str(tok).strip().partition("-")
        if not lo.isdigit() or not (hi or lo).isdigit() or int(hi or lo) < int(lo):
            raise SystemExit(f"--sim_port {spec!r}: expected ports such as 10-14 or 10,11")
        ports.extend(range(int(lo), int(hi or lo) + 1))
    return ports


# The Slurm submit node, resolved in the job: where the tunnel or relay to the sim servers ends.
SUBMIT_HOST = '"${SLURM_SUBMIT_HOST:-127.0.0.1}"'


def _packs[T](tasks: list[T], size: int) -> list[list[T]]:
    """``tasks`` in packs of ``size``: one job per pack, one worker per task."""
    return [tasks[i : i + size] for i in range(0, len(tasks), size)]


def _robotwin(
    common: dict[str, object],
    out: Path,
    num_workers: int,
    difficulties: tuple[str, ...],
    sim_host: str,
    sim_port: str | int | tuple[int, ...] | None,
    tasks: list[str],
) -> list[EvalJob]:
    prefix = f"env {_client_env({})} {_python('default')} {EVAL_ROLLOUT}"
    jobs = []
    if sim_port is not None:
        if len(difficulties) != 1:
            raise SystemExit(
                "--sim_port: the servers run one task config; evaluate one scene "
                "(task robotwin_easy or robotwin_hard)"
            )
        host = sim_host if sim_host == SUBMIT_HOST else shlex.quote(sim_host)
        prefix += f" --sim_host={host}"
        # The servers already run: one job per num_workers ports, one worker per port,
        # the tasks dealt round-robin over the jobs.
        ports = parse_ports(sim_port)
        size = min(num_workers, len(ports))
        if len(ports) % size:
            raise SystemExit(
                f"--sim_port: {len(ports)} ports do not divide into groups of "
                f"num_workers={num_workers} (one server per worker)"
            )
        groups = _packs(ports, size)
        d = difficulties[0]
        for i, group in enumerate(groups):
            pack = tasks[i :: len(groups)]
            if not pack:
                continue
            name = f"robotwin/{d}" if len(groups) == 1 else f"robotwin/{d}/pack{i:02d}"
            jobs.append((name, d, pack, len(group), {"sim_port": ",".join(map(str, group))}))
    else:
        base = _sim("robotwin", "RoboTwin/task_config", "mamba/envs/robotwin")
        mamba = _micromamba("robotwin")
        task_config = base / "RoboTwin/task_config"
        for d in difficulties:
            if d in CLEAN2RAND_KEYS:
                _derive_clean2rand_config(task_config, d)
            yml = (task_config / f"{ROBOTWIN_TASK_CONFIG[d]}.yml").read_text()
            for cam in ("head", "wrist"):
                if not re.search(rf"(?m)^\s*{cam}_camera_type:\s*Large_D435\s*$", yml):
                    raise SystemExit(
                        f"{task_config}/{ROBOTWIN_TASK_CONFIG[d]}.yml: {cam}_camera_type must be "
                        "Large_D435 (the training data is 640x480; scripts/eval/SETUP.md)"
                    )
            server = (
                f"PYTHONNOUSERSITE=1 PYTHONPATH= NVIDIA_DRIVER_CAPABILITIES=all "
                f"CUDA_HOME={base}/mamba/envs/robotwin XDG_CACHE_HOME={{portfile}}.mamba_cache "
                f"{mamba} run -r {base}/mamba -n robotwin python -s "
                f"{REPO}/scripts/eval/robotwin_sim_server.py --repo {base}/RoboTwin "
                f"--task-config {ROBOTWIN_TASK_CONFIG[d]} "
                f"--port 0 --port-file {{portfile}} > {{portfile}}.log 2>&1"
            )
            for i, pack in enumerate(_packs(tasks, num_workers)):
                sim = {"sim_server_cmd": server}
                jobs.append((f"robotwin/{d}/pack{i:02d}", d, pack, len(pack), sim))
    return [
        EvalJob(
            name,
            prefix,
            {
                "env_type": "robotwin",
                "task": ",".join(pack),
                **common,
                "num_workers": workers,
                **sim,
                "log_dir": out / f"robotwin_{d}",
            },
            scene=d,
        )
        for name, d, pack, workers, sim in jobs
    ]


def _robocasa(
    task: str, common: dict[str, object], out: Path, num_workers: int, tasks: list[str]
) -> list[EvalJob]:
    base = _sim(task, f"mamba/envs/{task}")
    mamba = _micromamba(task)
    env = "PYTHONNOUSERSITE=1 PYTHONPATH= XDG_CACHE_HOME={portfile}.mamba_cache"
    script = f"{REPO}/scripts/eval/robocasa365_sim_server.py"
    run = f"{mamba} run -r {base}/mamba -n {task} python -s {script}"
    # numba's LLVM JIT segfaults next to the preloaded Mesa llvmpipe LLVM: the server runs
    # without the JIT.
    server = (
        f"{env} NUMBA_DISABLE_JIT=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa "
        f"LD_PRELOAD={_osmesa()} {run} --port 0 --port-file {{portfile}} >> {{portfile}}.log 2>&1"
    )
    prefix = f"env {_client_env({})} {_python('default')} {EVAL_ROLLOUT}"
    # The packs stay within one suite.
    groups = {s: [t for t in tasks if SUITE_OF[t] == s] for s in ROBOCASA365_SUITES}
    return [
        EvalJob(
            f"{task}/{suite}/pack{i:02d}",
            prefix,
            {
                "env_type": task,
                "task": ",".join(pack),
                **common,
                "num_workers": len(pack),
                "sim_server_cmd": server,
                "log_dir": out / task,
            },
        )
        for suite, group in groups.items()
        for i, pack in enumerate(_packs(group, num_workers))
    ]


def eval_jobs(
    task: str,
    ckpt_dir: str,
    out_dir: str,
    *,
    use_ema: bool,
    n_episodes: int | None = None,
    num_workers: int = 6,
    history_k: int | None = None,
    n_action_steps: int | None = None,
    resume: bool = False,
    save_videos: bool = False,
    tasks: str | tuple[str, ...] | None = None,
    difficulties: str | tuple[str, ...] | None = None,
    sim_host: str = "127.0.0.1",
    sim_port: str | int | tuple[int, ...] | None = None,
) -> list[EvalJob]:
    """The rollout.py jobs of ``task`` on ``ckpt_dir``; results under ``out_dir``.
    ``n_episodes``: episodes per unit (``N_EPISODES[task]`` when None). ``num_workers``: rollout
    workers per job (socket benchmarks: tasks per pack, one worker each). ``history_k``: history
    window (the checkpoint's when None).
    ``tasks``: comma list restricting the units (LIBERO suites, or RoboTwin / RoboCasa365 task
    names). ``difficulties``: comma list of RoboTwin difficulties or Clean2Rand axes.
    ``sim_port``: RoboTwin servers already running on ``sim_host`` (``10-14``, ``10,11``).
    ``n_action_steps``: actions executed per predict (rollout.py's default when None; a K>1
    checkpoint forces its history stride)."""
    if task not in EVAL_BENCH_DIRS:
        raise SystemExit(f"unknown task {task!r} (one of {', '.join(EVAL_BENCH_DIRS)})")
    common = _common_args(
        ckpt_dir,
        n_episodes or N_EPISODES[task],
        use_ema=use_ema,
        resume=resume,
        save_videos=save_videos,
        history_k=history_k,
        n_action_steps=n_action_steps,
    )
    out = Path(out_dir)
    names = _comma_tuple(tasks)
    if task in ("libero", "libero_plus"):
        suites = names or LIBERO_SUITES
        if unknown := set(suites) - set(LIBERO_SUITES):
            raise SystemExit(f"unknown LIBERO suites {sorted(unknown)}")
        build = _libero if task == "libero" else _libero_plus
        return build(common, out, num_workers, suites)
    env_type = TASK_BENCHMARK[task]
    units = socket_bench_tasks(env_type, list(names) if names else None, ckpt_dir)
    if task in ("robotwin", "robotwin_clean2rand"):
        allowed = ("easy", "hard") if task == "robotwin" else tuple(CLEAN2RAND_KEYS)
        chosen = _comma_tuple(difficulties) or allowed
        if unknown := set(chosen) - set(allowed):
            raise SystemExit(f"{task}: unknown difficulties {sorted(unknown)} (have {allowed})")
        return _robotwin(common, out, num_workers, chosen, sim_host, sim_port, units)
    if sim_port is not None:
        raise SystemExit(f"--sim_port applies to robotwin tasks, not {task}")
    if difficulties:
        raise SystemExit(f"--difficulties applies to robotwin tasks, not {task}")
    if unknown := set(units) - set(SUITE_OF):
        raise SystemExit(f"robocasa365: unknown tasks {sorted(unknown)}")
    return _robocasa(task, common, out, num_workers, units)


def _comma_tuple(value: str | tuple[str, ...] | None) -> tuple[str, ...] | None:
    """Fire parses ``--tasks a,b`` as a tuple but ``--tasks a`` as a str."""
    if not value:
        return None
    return tuple(value) if isinstance(value, tuple) else tuple(value.split(","))


def check_bools(**flags: object) -> list[bool]:
    """The flags as bools (``fire_bool``). fire reads ``--resume submit`` as
    ``resume="submit"`` (the chained ``submit`` does not run): stop with the spelling that works."""
    try:
        return [fire_bool(name, value) for name, value in flags.items()]
    except ValueError as e:
        raise SystemExit(str(e)) from None


def main(
    task: str,
    ckpt_dir: str,
    out_dir: str,
    ema: bool = True,
    n_episodes: int | None = None,
    num_workers: int = 6,
    tasks: str | None = None,
    difficulties: str | None = None,
    sim_host: str = "127.0.0.1",
    sim_port: str | None = None,
    history_k: int | None = None,
    n_action_steps: int | None = None,
    resume: bool = False,
    save_videos: bool = False,
) -> None:
    """Print the eval commands of ``task`` (run them in order, one GPU each). ``--ema=False``
    evaluates the raw weights of a training checkpoint (an export has the EMA merged)."""
    ema, resume, save_videos = check_bools(ema=ema, resume=resume, save_videos=save_videos)
    task, difficulties = expand_short_task(task, difficulties)
    for job in eval_jobs(
        task,
        ckpt_dir,
        out_dir,
        use_ema=ema,
        n_episodes=n_episodes,
        num_workers=num_workers,
        tasks=tasks,
        difficulties=difficulties,
        sim_host=sim_host,
        sim_port=sim_port,
        history_k=history_k,
        n_action_steps=n_action_steps,
        resume=resume,
        save_videos=save_videos,
    ):
        print(job.command())


if __name__ == "__main__":
    import fire

    fire.Fire(main)
