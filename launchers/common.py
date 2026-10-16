"""Slurm submission helpers shared by launchers/sft.py and launchers/eval.py

Cluster settings come from the environment:

    REKI_RUNS_DIR         root of the run directories (<root>/<tag>/<hash>/{train,eval});
                          default <repo>/storage/runs (scripts/load_env.sh). Each kept
                          checkpoint is ~13 GB with EMA.
    REKI_GPUS_PER_NODE    GPUs per node (optional; default 8). A job with more GPUs takes
                          whole nodes.
    REKI_CPUS_PER_GPU     CPUs per GPU (optional; default 16, the released runs)
    REKI_EVAL_MEM_PER_GPU host RAM per eval job, e.g. 240G (optional; Slurm default otherwise)
    REKI_CONTAINER_IMAGE, REKI_CONTAINER_MOUNTS   pyxis container (optional)

Jobs inherit the submit shell's environment and read .env again on the compute node
(scripts/load_env.sh); eval commands bake in the sim paths of launchers/eval_protocols.py.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import re
import shutil
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from os.path import join as osp
from typing import Any

import yaml
from eval_protocols import (
    SUBMIT_HOST,
    TASK_SHORT_FORMS,
    check_bools,
    eval_jobs,
    expand_short_task,
    parse_ports,
)
from launcheon import SlurmExperimentGrid
from launcheon.experiment import Experiment
from launcheon.parser_format import ArgparseParserFormat, ParserFormat
from rich.console import Console
from rich.table import Table

from reki.eval import sim_protocol
from reki.eval.protocols import CLEAN2RAND_SCENE_DIRS, EVAL_BENCH_DIRS, ROBOTWIN_TASK_CONFIG
from reki.eval.results import (
    clean2rand_cells,
    collect,
    expected_units,
    result_cells,
    summarize,
)

REPO = pathlib.Path(__file__).resolve().parent.parent
OUTPUT_DIR = osp("{{launcheon.exp.log_dir}}", "train")


def runs_root() -> pathlib.Path:
    """The root of the run directories (created if missing)."""
    root = pathlib.Path(os.environ.get("REKI_RUNS_DIR") or REPO / "storage" / "runs").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _pixi_activation(env: str = "default") -> pathlib.Path:
    """Write the ``pixi shell-hook`` output of this checkout to ``.pixi/`` once, on the submit
    node, and return its path. Jobs source it and do not call pixi on the compute node. A
    change to ``pixi.lock`` or ``pyproject.toml`` gives a new file."""
    sha = hashlib.sha1()
    for name in ("pixi.lock", "pyproject.toml"):
        sha.update((REPO / name).read_bytes())
    lock = sha.hexdigest()[:8]
    path = REPO / ".pixi" / f"activate_{env}_{lock}.sh"
    if path.exists():
        return path
    pixi = shutil.which("pixi")
    if pixi is None:
        raise SystemExit(f"{path} is missing and pixi is not on PATH to write it")
    path.write_text(
        subprocess.check_output(
            [pixi, "shell-hook", "--frozen", "-e", env], cwd=str(REPO), text=True
        )
    )
    return path


def train_base_cmd(config: str) -> str:
    """The shell command prefix for training with ``config``."""
    launch = "accelerate launch --multi_gpu --mixed_precision=bf16 --dynamo_backend=no"
    return (
        f"source {_pixi_activation()} && export HF_HUB_OFFLINE=1 && "
        f"{launch} scripts/train.py --config={config} --resume=True"
    )


class FlagEqualsValue(ArgparseParserFormat):
    """``--flag=value``, which fire and argparse both parse. Training values are not quoted, so
    dict values keep the quotes that ``as_cli`` adds. Eval values are quoted because they hold
    spaces (``perturbation_axis``) and whole shell commands (``sim_server_cmd``)."""

    def __init__(self, quote_strings: bool = False):
        ParserFormat.__init__(
            self,
            kwargs_formatting="--{flag}={value}",
            repeat_key_in_repeated_args=True,
            quote_strings=quote_strings,
            none_value="",
        )


def _cluster_kwargs(quote_strings: bool = False) -> dict[str, object]:
    """Grid kwargs of every job: CPUs, the pyxis container when ``REKI_CONTAINER_IMAGE`` is
    set, and no partition, so ``sbatch`` uses ``SBATCH_PARTITION`` or the cluster default
    (``submit --partition`` overrides it)."""
    kwargs: dict[str, object] = dict(
        use_hash_in_dirnames=True,
        num_cpus_per_gpu=int(os.environ.get("REKI_CPUS_PER_GPU", 16)),
        partition=None,
        git_sync=False,
        work_dir=str(REPO),
        parser_format=FlagEqualsValue(quote_strings),
    )
    if image := os.environ.get("REKI_CONTAINER_IMAGE"):
        kwargs["container_image"] = image
        kwargs["container_mounts"] = os.environ.get("REKI_CONTAINER_MOUNTS")
    return kwargs


def grid_kwargs(*, total_gpus: int) -> dict[str, object]:
    """Constructor kwargs of the training grids: one node, or whole nodes of
    ``REKI_GPUS_PER_NODE`` GPUs."""
    per_node = int(os.environ.get("REKI_GPUS_PER_NODE", 8))
    if total_gpus > per_node and total_gpus % per_node:
        raise ValueError(f"{total_gpus} GPUs is not a multiple of {per_node} per node")
    return dict(
        num_nodes=max(1, total_gpus // per_node),
        num_gpus=min(total_gpus, per_node),
        locked=True,
        **_cluster_kwargs(),
    )


def as_cli(overrides: dict[str, object]) -> dict[str, object]:
    """Grid kwargs from TrainConfig overrides: a dict value goes as one quoted Python
    literal (``fire`` parses it back); launcheon would split it otherwise."""
    return {k: f'"{v!r}"' if isinstance(v, dict) else v for k, v in overrides.items()}


def runs_dir(tag: str) -> str:
    """The run directory for a launcher tag."""
    return str(runs_root() / tag)


def setup_label(exp: Experiment) -> str:
    """An experiment's setup label (``exp_<idx>`` without a setup group)."""
    groups = exp.kwargs_groups or {}
    return groups["setup"][0] if "setup" in groups else f"exp_{exp.idx}"


def _parse_steps(step: int | str | tuple[int, ...]) -> list[tuple[int, int]]:
    """``20000`` / ``20000,25000`` / ``15000-25000`` -> inclusive (lo, hi) ranges."""
    parts = step if isinstance(step, tuple) else str(step).split(",")
    ranges = []
    for part in parts:
        lo, _, hi = str(part).strip().partition("-")
        ranges.append((int(lo), int(hi or lo)))
    return ranges


def _select_steps(available: list[int], step: int | str | tuple[int, ...] | None) -> list[int]:
    if step is None:
        return [max(available)] if available else []
    if step == "all":
        return sorted(available)
    ranges = _parse_steps(step)
    return sorted(s for s in available if any(lo <= s <= hi for lo, hi in ranges))


def config_history_k(config: pathlib.Path) -> int | None:
    """The ``history_k`` of a ``train_config.yaml`` (``TrainConfig`` default: 3); None if absent."""
    if not config.exists():
        return None
    return yaml.safe_load(config.read_text()).get("history_k", 3)


def _checkpoint_steps(root: pathlib.Path) -> dict[int, pathlib.Path]:
    if not root.is_dir():
        return {}
    return {
        int(m.group(1)): d
        for d in root.iterdir()
        if (m := re.fullmatch(r"checkpoint-(\d+)", d.name))
    }


def warm_hf_cache(model_id: str) -> None:
    """Download a Hub model on the submit node. The jobs run with ``HF_HUB_OFFLINE=1``: they
    read it from the cache and send no request to the Hub."""
    from huggingface_hub import snapshot_download

    if not os.path.isdir(model_id):
        snapshot_download(model_id)


def _listening(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def ensure_sim_relay(port: int, log_dir: pathlib.Path) -> None:
    """Forward ``port`` of this node's cluster address to its loopback ``port``, where the sim
    tunnel ends (scripts/eval/sim_relay.py). Eval jobs connect to the cluster address. No-op
    when the relay is already up. The relay keeps running after the submit."""
    host = socket.gethostbyname(socket.gethostname())
    if _listening(host, port):
        print(f"[relay] {host}:{port} already up")
        return
    if not _listening("127.0.0.1", port):
        raise SystemExit(f"--sim_port {port}: no sim server or tunnel on 127.0.0.1:{port}")
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"sim_relay.{port}.log"
    with open(log, "ab") as f:
        subprocess.Popen(
            ["python3", str(REPO / "scripts/eval/sim_relay.py"), str(port), str(port)],
            stdin=subprocess.DEVNULL,
            stdout=f,
            stderr=f,
            start_new_session=True,
        )
    for _ in range(20):
        if _listening(host, port):
            print(f"[relay] started {host}:{port} -> 127.0.0.1:{port} (log {log})")
            return
        time.sleep(0.25)
    raise SystemExit(f"[relay] {host}:{port} did not come up, see {log}")


# A busy server answers only when its current client leaves.
SIM_CHECK_TIMEOUT_S = 300


def _sim_server_problem(port: int, task_config: str, task: str) -> str | None:
    """None if the RoboTwin server behind the loopback ``port`` makes ``task`` with
    ``task_config``, else what is wrong."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=10) as conn:
            conn.settimeout(SIM_CHECK_TIMEOUT_S)
            reply = sim_protocol.rpc(conn, {"cmd": "make", "task": task})
            sim_protocol.send(conn, {"cmd": "close"})
    except TimeoutError:
        return f"no reply in {SIM_CHECK_TIMEOUT_S} s (busy with another client, or hung)"
    except (OSError, ConnectionError) as e:
        return f"no sim server ({e})"
    except sim_protocol.SimServerError as e:
        return f"make {task} failed: {e.error}"
    served = reply.get("code", {}).get("task_config")
    if served != task_config:
        return f"serves {served}"
    return None


def check_sim_servers(ports: tuple[int, ...], task_config: str, task: str) -> None:
    """Send a ``make`` to each RoboTwin server on the loopback ``ports``. A tunnel stays open
    when the server behind it is down, so a listening port does not prove a server."""
    with ThreadPoolExecutor(len(ports)) as pool:
        problems = pool.map(lambda p: _sim_server_problem(p, task_config, task), ports)
        bad = {p: why for p, why in zip(ports, problems) if why}
    if bad:
        lines = "\n".join(f"  {p}: {why}" for p, why in bad.items())
        raise SystemExit(f"--sim_port: expected {task_config}\n{lines}")
    print(f"[sim] {len(ports)} servers serve {task_config}")


class JobGrid(SlurmExperimentGrid):
    """``submit`` downloads ``vlm_model_ids`` into the HF cache and starts the ``relay_ports``
    sim relays; ``submit --partition <name>`` sends the jobs to that Slurm partition."""

    vlm_model_ids: tuple[str, ...] = ()
    relay_ports: tuple[int, ...] = ()
    relay_log_dir: pathlib.Path | None = None
    # (task config, task) the relay_ports servers must serve.
    sim_check: tuple[str, str] | None = None
    # `eval --resume` also resumes the runs whose log dir exists (launcheon's submit --resume).
    resume: bool = False

    def submit(
        self,
        *args: Any,
        partition: str | None = None,
        resume: bool | None = None,
        **kwargs: Any,
    ) -> None:
        for model_id in self.vlm_model_ids:
            warm_hf_cache(model_id)
        for port in self.relay_ports:
            assert self.relay_log_dir is not None
            ensure_sim_relay(port, self.relay_log_dir)
        if self.sim_check:
            check_sim_servers(self.relay_ports, *self.sim_check)
        if partition:
            self.sbatch_kwargs["partition"] = partition
        super().submit(*args, resume=self.resume if resume is None else resume, **kwargs)


# Non-RoboTwin benchmarks whose units are tasks: they get a per-task table.
TASK_UNIT_BENCHES = ("robocasa365",)


def _task_rates(out: pathlib.Path, bench: str) -> dict[str, str]:
    """Success rate (%) per task of ``bench`` under ``out``; "-" for the tasks not run yet, grey
    for the tasks short of episodes."""
    done = collect(out, (bench,))
    summary = summarize(done)[bench]
    missing = [t for g in summary["groups"].values() for t in g["missing_units"]]
    missing += [t for g in summary["missing_groups"] for t in expected_units(bench, g, []) or ()]
    rates = dict.fromkeys(missing, "-")
    for u in (u for units in done[bench].values() for u in units):
        rate = f"{100 * u['n_success'] / u['n_episodes']:.1f}"
        rates[u["unit"]] = f"[dim]{rate}[/dim]" if u["partial"] else rate
    return rates


class RekiGrid(JobGrid):
    """launcheon grid with two eval verbs on the selected runs:

    * ``eval``: build a 1-GPU job grid running the protocol of ``launchers/eval_protocols.py``
      on ``<run>/train/checkpoint-<step>``; chain ``submit`` / ``table`` / ``cmd``.
    * ``results``: tabulate the results already under ``<run>/eval``.

    ``default_eval_tasks`` maps a setup label to its default eval task.
    """

    default_eval_tasks: dict[str, str] = {}
    row_column = "step"
    checkpoint_variants = True
    table_columns = "setup"
    config = ""

    def preflight(self) -> None:
        """Check the datasets of the selected runs and download their models into the HF
        cache, on the submit node."""
        from reki.data.robotics.registry import expand_spec
        from reki.policy.checkpoint import resolve_checkpoint

        for exp in self:
            try:
                expand_spec(str(exp.kwargs["dataset"]))
            except FileNotFoundError as e:
                raise SystemExit(str(e)) from None
            except RuntimeError as e:
                raise SystemExit(f"{e}\nsee docs/data.md#benchmark-datasets") from None
        for path in {str(p) for exp in self if (p := exp.kwargs.get("pretrained_path"))}:
            resolve_checkpoint(path)
        warm_hf_cache(yaml.safe_load((REPO / self.config).read_text())["vlm_model_id"])

    def submit(self, *args: Any, **kwargs: Any) -> None:
        self.preflight()
        super().submit(*args, **kwargs)

    def table(self, *args: Any, columns: str | None = None, **kwargs: Any) -> None:
        super().table(*args, columns=columns or self.table_columns, **kwargs)

    def _task_for(self, exp: Experiment, task: str | None) -> str:
        task = task or self.default_eval_tasks.get(setup_label(exp))
        if task is None:
            raise SystemExit(f"pass --task (one of {', '.join(EVAL_BENCH_DIRS)})")
        if task not in EVAL_BENCH_DIRS:
            names = ", ".join([*EVAL_BENCH_DIRS, *TASK_SHORT_FORMS])
            raise SystemExit(f"unknown task {task!r} (one of {names})")
        return task

    def eval(
        self,
        *,
        task: str | None = None,
        step: int | str | tuple[int, ...] | None = None,
        ema: bool = True,
        n_episodes: int | None = None,
        num_workers: int = 6,
        history_k: int | None = None,
        n_action_steps: int | None = None,
        resume: bool = False,
        save_videos: bool = False,
        tasks: str | tuple[str, ...] | None = None,
        difficulties: str | tuple[str, ...] | None = None,
        sim_host: str = SUBMIT_HOST,
        sim_port: str | int | tuple[int, ...] | None = None,
        sim_relay: bool = False,
    ) -> SlurmExperimentGrid:
        """1-GPU eval grid (launchers/eval_protocols.py) for the selected runs.

        ``step``: ``20000``, ``20000,25000``, ``15000-25000`` or ``all`` (default: the latest
        checkpoint). ``ema=False`` evaluates the raw weights, not the EMA shadow. ``tasks`` and
        ``difficulties`` restrict the units (see ``eval_jobs``). ``robotwin_easy`` is
        ``robotwin`` with ``difficulties=easy``. ``sim_port``: ports of RoboTwin servers that
        already run on ``sim_host`` (e.g. ``10-14``), one worker per port. ``sim_host``
        defaults to the submit node. ``history_k``: frames in the window; another window than the
        trained one writes to a ``k<K>`` subdir. ``n_action_steps``: actions executed per
        predict; its results go to an ``n<N>`` subdir. The arguments are keyword-only, so fire chains a trailing
        ``submit`` / ``table`` / ``cmd`` onto the returned grid."""
        ema, resume, save_videos, sim_relay = check_bools(
            ema=ema, resume=resume, save_videos=save_videos, sim_relay=sim_relay
        )
        # robotwin_easy / robotwin_hard keep their own job and log names.
        label = task if task in TASK_SHORT_FORMS else None
        task, difficulties = expand_short_task(task, difficulties)
        bench_tasks = {self._task_for(exp, task) for exp in self}
        if len(bench_tasks) != 1:
            raise SystemExit(f"the selection evaluates on {sorted(bench_tasks)}: pass --task")
        (task_,) = bench_tasks
        targets = []
        for exp in self:
            exp_targets = self._eval_targets(exp, step, ema)
            if history_k not in (None, self._trained_history_k(exp)):
                # Another window: its own results, next to those of the trained window.
                exp_targets = [
                    (f"{label}_k{history_k}", c, osp(o, f"k{history_k}"))
                    for label, c, o in exp_targets
                ]
            targets += exp_targets
        if not targets:
            raise SystemExit(f"no selected run has a checkpoint matching step={step}")
        if n_action_steps is not None:
            targets = [
                (f"{label}_n{n_action_steps}", c, osp(o, f"n{n_action_steps}"))
                for label, c, o in targets
            ]
        if sim_port is not None and len(targets) > 1:
            raise SystemExit(
                f"--sim_port: {len(targets)} checkpoints would share the servers, and a server "
                "serves one client at a time: evaluate one checkpoint (select, --step)"
            )
        if sim_port is not None and sim_host == SUBMIT_HOST and not sim_relay:
            ports = ",".join(map(str, parse_ports(sim_port)))
            raise SystemExit(
                f"--sim_port {ports} on the default host requires either --sim_relay=True, "
                "which exposes an unauthenticated pickle port on <cluster address>:<port> "
                "until killed with `pkill -f scripts/eval/sim_relay.py`, or ports you make "
                "reachable yourself with --sim_host <address>"
            )
        return self._job_grid(
            task_,
            targets,
            name=label,
            use_ema=ema,
            n_episodes=n_episodes,
            num_workers=num_workers,
            history_k=history_k,
            n_action_steps=n_action_steps,
            resume=resume,
            save_videos=save_videos,
            tasks=tasks,
            difficulties=difficulties,
            sim_host=sim_host,
            sim_port=sim_port,
        )

    def _eval_targets(
        self, exp: Experiment, step: int | str | tuple[int, ...] | None, ema: bool
    ) -> list[tuple[str, str, str]]:
        """(job label, checkpoint dir, results dir) of each checkpoint of ``exp`` to evaluate."""
        ckpts = _checkpoint_steps(pathlib.Path(exp.log_dir) / "train")
        steps = _select_steps(list(ckpts), step)
        if not steps:
            print(
                f"{setup_label(exp)}: no checkpoint matches step={step} "
                f"(on disk: {sorted(ckpts) or 'none'}), skipped"
            )
            return []
        targets = []
        for s in steps:
            if ema and not (ckpts[s] / "ema.safetensors").exists():
                raise SystemExit(f"{ckpts[s]} has no ema.safetensors; pass --ema=False")
            out = osp(exp.log_dir, "eval", f"checkpoint-{s}", "ema" if ema else "raw")
            targets.append((f"{setup_label(exp)}_{s // 1000}k", str(ckpts[s]), out))
        return targets

    def _trained_history_k(self, exp: Experiment) -> int | None:
        """The ``history_k`` ``exp`` was trained with; None if unknown."""
        return config_history_k(pathlib.Path(exp.log_dir) / "train" / "train_config.yaml")

    def _result_dirs(
        self, exp: Experiment, step: int | str | tuple[int, ...] | None, ema: bool
    ) -> list[tuple[str, pathlib.Path]]:
        """(row label, results dir) of each evaluated checkpoint of ``exp``."""
        evals = _checkpoint_steps(pathlib.Path(exp.log_dir) / "eval")
        variant = "ema" if ema else "raw"
        return [(str(s), evals[s] / variant) for s in _select_steps(list(evals), step)]

    def _job_grid(
        self,
        task: str,
        targets: list[tuple[str, str, str]],
        name: str | None = None,
        **eval_kwargs: object,
    ) -> JobGrid:
        """1-GPU grid of the ``eval_jobs`` of ``task`` for each (label, ckpt, out dir)."""
        prefixes: set[str] = set()
        setup: dict[str, dict[str, str]] = {}
        scenes: set[str | None] = set()
        for label, ckpt, out in targets:
            for job in eval_jobs(task, ckpt, out, **eval_kwargs):  # type: ignore[arg-type]
                prefixes.add(job.prefix)
                scenes.add(job.scene)
                # --resume goes in the shared prefix, out of the run hash: a resumed job keeps
                # its run (and its status, for `select unscheduled`).
                setup[f"{label}_{job.name.replace('/', '_')}"] = {
                    k: str(v) for k, v in job.args.items() if k != "resume"
                }
        (prefix,) = prefixes
        resume = bool(eval_kwargs.get("resume"))
        if resume:
            prefix += " --resume=True"
        grid = JobGrid(
            base_name=f"{self.base_name}_eval_{name or task}",
            base_cmd=f"export HF_HUB_OFFLINE=1 && {prefix}",
            log_dir=f"{self.log_dir_root}_eval",
            num_nodes=1,
            num_gpus=1,
            setup=setup,
            **_cluster_kwargs(quote_strings=True),  # type: ignore[arg-type]
        )
        # reki.policy imports torch (~1.5 s): only eval needs it.
        from reki.policy.train_config import checkpoint_config_path, read_checkpoint_config

        configs = [checkpoint_config_path(ckpt) for _, ckpt, _ in targets]
        grid.vlm_model_ids = tuple(
            {read_checkpoint_config(c)["vlm_model_id"] for c in configs if c is not None}
        )
        grid.resume = resume
        if mem := os.environ.get("REKI_EVAL_MEM_PER_GPU"):
            grid.sbatch_kwargs["mem-per-gpu"] = mem
        sim_port = eval_kwargs.get("sim_port")
        if sim_port is not None and eval_kwargs.get("sim_host") == SUBMIT_HOST:
            grid.relay_ports = tuple(parse_ports(sim_port))  # type: ignore[arg-type]
            grid.relay_log_dir = pathlib.Path(targets[0][2])
            first = next(iter(setup.values()))
            (scene,) = scenes  # eval_jobs: --sim_port evaluates one scene
            assert scene is not None
            grid.sim_check = (
                ROBOTWIN_TASK_CONFIG[scene],
                first["task"].split(",")[0],
            )
        return grid

    def results(
        self,
        *,
        task: str | None = None,
        step: int | str | tuple[int, ...] | None = "all",
        ema: bool = True,
        history_k: int | tuple[int, ...] | None = None,
        n_action_steps: int | None = None,
    ) -> None:
        """Success rates (%) per run x checkpoint from ``<run>/eval``, the mean env steps of the
        successful episodes, and the episodes done / expected. Grey cells have missing episodes
        or units. RoboTwin: one row per difficulty; Clean2Rand adds Easy,
        Hard and the mean of the 6 scenes (bold). RoboTwin and RoboCasa: under a per-task table.
        ``history_k``, ``n_action_steps``: the results of ``eval --history_k`` /
        ``eval --n_action_steps``. ``history_k=1,2``: one row per window; the trained window
        reads the results of ``eval`` without ``--history_k``."""
        (ema,) = check_bools(ema=ema)
        variant = "ema" if ema else "raw"
        rows: list[tuple[str, str, str, str, dict[str, str]]] = []
        columns: dict[str, None] = {}
        # Per-task success rates (%) of the benchmarks whose units are tasks:
        # {(id, step, difficulty): {task: cell}}.
        per_task: dict[tuple[str, str, str], dict[str, str]] = {}
        ks = (history_k,) if isinstance(history_k, int) else history_k
        k_rows = ks is not None and len(ks) > 1
        for exp in self:
            clean2rand = False
            if task in TASK_SHORT_FORMS:
                bench, difficulty = TASK_SHORT_FORMS[task]
                self._task_for(exp, bench)
                bench_dirs = (f"robotwin_{difficulty}",)
            else:
                bench_dirs = EVAL_BENCH_DIRS[self._task_for(exp, task)]
                clean2rand = self._task_for(exp, task) == "robotwin_clean2rand"
            trained_k = self._trained_history_k(exp) if ks else None
            for row_label, step_out in self._result_dirs(exp, step, ema):
                for k in ks or (None,):
                    out = step_out if k in (None, trained_k) else step_out / f"k{k}"
                    if n_action_steps is not None:
                        out = out / f"n{n_action_steps}"
                    if not out.is_dir():
                        continue
                    key = (str(exp.idx), row_label)
                    k_label = f"k{k}" if k_rows else ""
                    if not bench_dirs[0].startswith("robotwin_"):
                        cells = result_cells(out, bench_dirs)
                        if cells:
                            columns.update(dict.fromkeys(cells))
                            rows.append((str(exp.idx), setup_label(exp), row_label, k_label, cells))
                            for bench in bench_dirs:
                                if bench in TASK_UNIT_BENCHES:
                                    per_task[(*key, k_label)] = _task_rates(out, bench)
                        continue
                    # Clean2Rand: the 6 scenes, then their mean.
                    dirs = CLEAN2RAND_SCENE_DIRS if clean2rand else bench_dirs
                    for bench in dirs:
                        cells = result_cells(out, (bench,))
                        if not cells:
                            continue
                        diff = " ".join(x for x in (bench.removeprefix("robotwin_"), k_label) if x)
                        cells["avg"] = cells.pop(bench)
                        columns.update(dict.fromkeys(cells))
                        rows.append((str(exp.idx), setup_label(exp), row_label, diff, cells))
                        per_task[(*key, diff)] = _task_rates(out, bench)
                    if clean2rand and (cells := clean2rand_cells(out, dirs)):
                        diff = " ".join(x for x in ("mean", k_label) if x)
                        rows.append((str(exp.idx), setup_label(exp), row_label, diff, cells))
        title = " ".join(x for x in (self.base_name, task, "results") if x)
        if ks:
            title += f" k={','.join(map(str, ks))}"
        if n_action_steps is not None:
            title += f" n_action_steps={n_action_steps}"
        if self.checkpoint_variants:
            title += f" ({variant})"
        if not rows:
            print(f"{title}: none on disk for the selection")
            return
        if per_task:
            tasks = sorted({t for cells in per_task.values() for t in cells})
            table = Table(title=f"{title}: per task (%)")
            table.add_column("#", justify="right")
            table.add_column("task")
            for idx, row_label, diff in per_task:
                label = f"{int(row_label) // 1000}k" if row_label.isdigit() else row_label
                table.add_column(
                    " ".join(x for x in (f"[{idx}]", label, diff) if x), justify="right"
                )
            for i, t in enumerate(tasks):
                table.add_row(f"{i:02d}", t, *(cells.get(t, "-") for cells in per_task.values()))
            Console().print(table)
        has_variant = any(diff for *_, diff, _ in rows)
        table = Table(title=title)
        heads = ("id", "setup", self.row_column, *(("variant",) if has_variant else ()), *columns)
        for name in heads:
            table.add_column(name, justify="left" if name in ("setup", "variant") else "right")
        for idx, label, row_label, diff, cells in rows:
            table.add_row(
                idx,
                label,
                row_label,
                *((diff,) if has_variant else ()),
                *(cells.get(c, "-") for c in columns),
            )
        Console().print(table)
