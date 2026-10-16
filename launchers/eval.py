"""Evaluation of exported checkpoints (HF model ids or local export dirs) on the released
protocols of launchers/eval_protocols.py. Results go under
$REKI_RUNS_DIR/models/<hash>/.

    pixi run python launchers/eval.py table
    pixi run python launchers/eval.py select 0 eval --task libero submit
    pixi run python launchers/eval.py select 0 eval --task libero_plus submit
    pixi run python launchers/eval.py select 0 results

To evaluate your own export, add a line to MODELS. Environment: see launchers/common.py.
"""

from __future__ import annotations

import copy
import json
import pathlib
from collections.abc import Callable
from typing import Any

from common import RekiGrid, config_history_k, runs_dir, setup_label
from eval_protocols import SUBMIT_HOST, TASK_SHORT_FORMS, check_bools, expand_short_task
from launcheon.experiment import Experiment

from reki.eval.protocols import EVAL_BENCH_DIRS, TASK_BENCHMARK

MODELS: dict[str, dict[str, Any]] = {
    "reki-2b-libero": {"ckpt": "kyutai/reki-2b-libero", "tasks": ["libero", "libero_plus"]},
    "reki-2b-robotwin-clean": {
        "ckpt": "kyutai/reki-2b-robotwin-clean",
        "tasks": ["robotwin", "robotwin_clean2rand"],
    },
    "reki-2b-robotwin-cleanrand": {
        "ckpt": "kyutai/reki-2b-robotwin-cleanrand",
        "tasks": ["robotwin"],
    },
    "reki-2b-robocasa365": {"ckpt": "kyutai/reki-2b-robocasa365", "tasks": ["robocasa365"]},
}

SOURCE_FILE = "source.json"


def resolve(ckpt: str) -> pathlib.Path:
    """A local export dir as is; an HF model id downloaded into HF_HOME."""
    from reki.policy.checkpoint import resolve_checkpoint

    path = resolve_checkpoint(ckpt, include_ema=True)
    if not (path / "reki_inference.json").exists():
        raise SystemExit(f"{ckpt} has no reki_inference.json: export it first")
    return path


class GridSet:
    """The job grids of several tasks, driven by one chained verb. ``build(submit)`` makes
    them; only ``submit`` downloads the checkpoints and records ``source.json``."""

    def __init__(self, build: Callable[[bool], list]) -> None:
        self.build = build

    def submit(self, *args: Any, **kwargs: Any) -> None:
        for grid in self.build(True):
            grid.submit(*args, **kwargs)

    def table(self, *args: Any, **kwargs: Any) -> None:
        for grid in self.build(False):
            grid.table(*args, **kwargs)

    def cmd(self) -> None:
        for grid in self.build(False):
            grid.cmd()

    def status(self) -> None:
        for grid in self.build(False):
            grid.status()


class ModelGrid(RekiGrid):
    """One experiment per MODELS entry. ``eval`` evaluates the export itself (no training
    steps); its results dir is the experiment dir."""

    row_column = "revision"
    checkpoint_variants = False
    table_columns = "setup,tasks"
    _submitting = False

    @staticmethod
    def _tasks(exp: Experiment) -> list[str]:
        return list(MODELS[setup_label(exp)]["tasks"])  # type: ignore[call-overload]

    def _task_for(self, exp: Experiment, task: str | None) -> str:
        if task is None or task not in self._tasks(exp):
            raise SystemExit(f"{setup_label(exp)} evaluates on {self._tasks(exp)}, not {task}")
        if task not in EVAL_BENCH_DIRS:
            raise SystemExit(f"unknown task {task!r} (one of {', '.join(EVAL_BENCH_DIRS)})")
        return task

    def _by_task(self, task: str | None) -> dict[str, ModelGrid]:
        """A copy of the selection per task, each holding the models that evaluate on it."""
        names = [task] if task else list(dict.fromkeys(t for e in self for t in self._tasks(e)))
        subs = {}
        for name in names:
            sub = copy.copy(self)
            sub._exps_dict = {k: e for k, e in self._exps_dict.items() if name in self._tasks(e)}
            if not sub._exps_dict:
                raise SystemExit(f"no selected model evaluates on {name}")
            subs[name] = sub
        return subs

    def _eval_targets(
        self, exp: Experiment, step: int | str | tuple[int, ...] | None, ema: bool
    ) -> list[tuple[str, str, str]]:
        if step is not None:
            raise SystemExit("an export has no training steps: drop --step")
        ckpt = str(MODELS[setup_label(exp)]["ckpt"])
        out = pathlib.Path(exp.log_dir)
        if not self._submitting:
            # The path of the last submit, so job hashes match; else the ckpt as given.
            source = out / SOURCE_FILE
            path_ = json.loads(source.read_text())["path"] if source.exists() else ckpt
            return [(setup_label(exp), path_, exp.log_dir)]
        path = resolve(ckpt)
        benchmark = json.loads((path / "reki_inference.json").read_text())["benchmark"]
        wanted = {TASK_BENCHMARK[t] for t in self._tasks(exp)}
        if wanted != {benchmark}:
            raise SystemExit(f"{ckpt} is exported for {benchmark!r}, its tasks need {wanted}")
        if ema and not (path / "ema.safetensors").exists():
            raise SystemExit(f"{path} has no ema.safetensors; drop --ema")
        out.mkdir(parents=True, exist_ok=True)
        (out / SOURCE_FILE).write_text(json.dumps({"ckpt": ckpt, "path": str(path)}) + "\n")
        return [(setup_label(exp), str(path), exp.log_dir)]

    def _trained_history_k(self, exp: Experiment) -> int | None:
        source = pathlib.Path(exp.log_dir) / SOURCE_FILE
        if not source.exists():
            return None
        path = pathlib.Path(json.loads(source.read_text())["path"])
        return config_history_k(path / "train_config.yaml")

    def _result_dirs(
        self, exp: Experiment, step: int | str | tuple[int, ...] | None, ema: bool
    ) -> list[tuple[str, pathlib.Path]]:
        if step is not None:
            raise SystemExit("an export has no training steps: drop --step")
        out = pathlib.Path(exp.log_dir)
        source = out / SOURCE_FILE
        if not source.exists():
            if next(out.rglob("results.json"), None) is None:
                return []
            raise SystemExit(
                f"{out} has results but no {SOURCE_FILE}: the checkpoint they came from is "
                "unknown (it is written by `eval submit`)"
            )
        path = json.loads(source.read_text())["path"]
        revision = pathlib.Path(path).name[:10] if "/snapshots/" in path else "local"
        return [(revision, out)]

    def submit(self, *args: Any, **kwargs: Any) -> None:
        raise SystemExit("use `select <i> eval submit`")

    def eval(
        self,
        *,
        task: str | None = None,
        step: int | str | tuple[int, ...] | None = None,
        ema: bool = False,
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
    ) -> GridSet:
        """1-GPU eval grids of the selected models, one per task (``--task`` restricts it).
        Other flags as in ``RekiGrid.eval``. There is no ``**kwargs``: fire would pass it the
        flags of a chained ``submit``."""
        ema, resume, save_videos, sim_relay = check_bools(
            ema=ema, resume=resume, save_videos=save_videos, sim_relay=sim_relay
        )
        task, difficulties = expand_short_task(task, difficulties)
        subs = self._by_task(task)

        def build(submit: bool) -> list:
            grids = []
            for name, sub in subs.items():
                sub._submitting = submit
                grids.append(
                    RekiGrid.eval(
                        sub,
                        task=name,
                        step=step,
                        ema=ema,
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
                        sim_relay=sim_relay,
                    )
                )
            return grids

        return GridSet(build)

    def results(
        self,
        *,
        task: str | None = None,
        step: int | str | tuple[int, ...] | None = None,
        ema: bool = False,
        history_k: int | tuple[int, ...] | None = None,
        n_action_steps: int | None = None,
    ) -> None:
        """Success rates (%) of the selected models, one table per task."""
        short = task if task in TASK_SHORT_FORMS else None
        for name, sub in self._by_task(TASK_SHORT_FORMS[short][0] if short else task).items():
            RekiGrid.results(
                sub,
                task=short or name,
                step=step,
                ema=ema,
                history_k=history_k,
                n_action_steps=n_action_steps,
            )


if __name__ == "__main__":
    tag = "models"
    grid = ModelGrid(
        base_name=tag,
        base_cmd="false",
        log_dir=runs_dir(tag),
        use_hash_in_dirnames=True,
        expand_groups_in_name=False,
        git_sync=False,
        setup={
            name: {"ckpt": m["ckpt"], "tasks": ",".join(m["tasks"])} for name, m in MODELS.items()
        },  # type: ignore[arg-type]
    )
    grid.fire()
