"""Success rates from the ``results.json`` files of ``scripts/eval/rollout.py``.

An ``<out_dir>`` has one dir per benchmark (``reki.eval.protocols.EVAL_BENCH_DIRS``). Each has
one leaf ``results.json`` per unit: a LIBERO (suite, task_id) or a socket-bench task.
``summarize(collect(out_dir))`` gives per group (suite, axis) the success rate and counts,
and per benchmark the score:

  * LIBERO: mean over the (suite, task) units
  * LIBERO-Plus: successes over all episodes, overall and per axis
  * RoboTwin: mean over tasks
  * RoboCasa365: the official macro over the 50 target tasks

A group or benchmark is ``partial`` when a unit has fewer episodes than its target or an
expected unit or group has no results.
"""

from __future__ import annotations

import functools
import json
import os
import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from reki.eval.protocols import EXPECTED_GROUPS, GROUP_SIZES, LIBERO_TASK_IDS, socket_bench_tasks
from reki.eval.robocasa365_protocol import SUITE_OF as RC365_SUITE_OF
from reki.eval.robocasa365_protocol import TARGET_SUITES as RC365_SUITES

Table = dict[str, dict[str, list[dict]]]


def _socket_env(bench: str) -> str | None:
    """The rollout env_type of a socket-bench result dir whose units are its tasks."""
    return "robotwin" if bench.startswith("robotwin_") else None


@functools.cache
def _task_grid(env_type: str, ckpt_dir: str) -> tuple[str, ...]:
    """The full task grid of a socket bench: one dataset scan per checkpoint."""
    return tuple(socket_bench_tasks(env_type, None, ckpt_dir))


def expected_units(bench: str, group: str, units: list[dict]) -> tuple[str, ...] | None:
    """The units the full protocol runs in ``group`` of ``bench`` (None: not checked)."""
    if bench == "libero":
        return tuple(f"{tid:02d}" for tid in LIBERO_TASK_IDS)
    if bench == "robocasa365":
        return tuple(RC365_SUITES.get(group, ()))
    env_type = _socket_env(bench)
    if env_type is None:
        return None
    grids = {_task_grid(env_type, ckpt_dir) for ckpt_dir in {u["ckpt_dir"] for u in units}}
    if len(grids) != 1:
        raise SystemExit(f"{bench}: the leaves' checkpoints disagree on the task list")
    return grids.pop()


def read_leaf(path: Path) -> dict[str, Any] | None:
    """A unit's results, or None when the file does not read as JSON (e.g. while a worker
    rewrites it)."""
    try:
        run = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return run


def _group_and_unit(bench: str, rel_parts: tuple[str, ...]) -> tuple[str, str]:
    """(group, unit) from the leaf's dir path relative to the benchmark dir."""
    unit = rel_parts[-1]
    if bench == "robocasa365":
        return RC365_SUITE_OF.get(unit, "all"), unit
    if len(rel_parts) == 1:
        return "all", unit
    return "/".join(rel_parts[:-1]), unit


# Leaf scans and reads are bound by file-server latency (many small files), not CPU.
_IO_THREADS = 32


def _find_leaves(roots: list[Path], pool: ThreadPoolExecutor) -> list[Path]:
    """Every ``results.json`` under ``roots`` (following symlinks), one directory level at a
    time with the levels' directories listed in parallel."""
    found: list[Path] = []
    level = roots
    while level:
        subdirs: list[Path] = []
        for entries in pool.map(lambda d: list(os.scandir(d)), level):
            for e in entries:
                if e.name == "results.json" and e.is_file():
                    found.append(Path(e.path))
                elif e.is_dir():
                    subdirs.append(Path(e.path))
        level = subdirs
    return found


def collect(out_dir: Path, benches: tuple[str, ...] | None = None) -> Table:
    """The leaves under ``out_dir``, or only under its ``benches`` dirs."""
    table: Table = defaultdict(lambda: defaultdict(list))
    roots = [out_dir] if benches is None else [out_dir / b for b in benches]
    with ThreadPoolExecutor(_IO_THREADS) as pool:
        paths = sorted(_find_leaves([r for r in roots if r.is_dir()], pool))
        runs = list(pool.map(read_leaf, paths))
    for path, run in zip(paths, runs):
        if run is None:
            continue
        rel = path.parent.relative_to(out_dir).parts
        if len(rel) < 2:
            continue
        bench, rest = rel[0], rel[1:]
        group, unit = _group_and_unit(bench, rest)
        ne, ns = run["n_episodes"], run["n_success"]
        target = run["target_episodes"]
        entry = {
            "unit": unit,
            "n_episodes": ne,
            "n_success": ns,
            "target": target,
            "partial": ne < target,
            "success_steps": [e["steps"] for e in run["episodes"] if e["success"]],
        }
        if _socket_env(bench):
            entry["ckpt_dir"] = run["ckpt_dir"]
        table[bench][group].append(entry)
    return table


def _pooled(units: list[dict]) -> float:
    """Successes over episodes: the standard LIBERO-Plus overall and per-axis score."""
    ne = sum(u["n_episodes"] for u in units)
    return sum(u["n_success"] for u in units) / ne if ne else float("nan")


def _macro(units: list[dict]) -> float:
    rates = [u["n_success"] / u["n_episodes"] for u in units if u["n_episodes"]]
    return statistics.fmean(rates) if rates else float("nan")


def summarize(table: Table) -> dict[str, dict]:
    """Per-group and per-benchmark scores aggregated from unit results."""
    out: dict[str, dict] = {}
    for bench, groups in sorted(table.items()):
        gsum: dict[str, dict] = {}
        for group, units in sorted(groups.items()):
            expected = expected_units(bench, group, units)
            missing = sorted(set(expected or ()) - {u["unit"] for u in units})
            unit_target = max(u["target"] for u in units)
            # Units not on disk yet: by name where known, else by count (LIBERO-Plus).
            n_missing = max(len(missing), GROUP_SIZES.get(bench, {}).get(group, 0) - len(units))
            gsum[group] = {
                "pooled": _pooled(units),
                "macro": _macro(units),
                "n_units": len(units),
                "n_episodes": sum(u["n_episodes"] for u in units),
                # Missing units count at the target of the others.
                "target_episodes": sum(u["target"] for u in units) + n_missing * unit_target,
                "missing_units": missing,
                "partial": n_missing > 0 or any(u["partial"] for u in units),
            }
        all_units = [u for units in groups.values() for u in units]
        if bench == "libero_plus":
            by_axis: dict[str, list[dict]] = defaultdict(list)
            for group, units in groups.items():
                suite, _, axis = group.partition("/")
                by_axis[axis or suite].extend(units)
            per_axis = {a: _pooled(v) for a, v in sorted(by_axis.items())}
            out[bench] = {"groups": gsum, "per_axis": per_axis, "macro": _pooled(all_units)}
        elif len(gsum) == 1 and "all" in gsum:
            out[bench] = {"groups": gsum, "macro": gsum["all"]["macro"]}
        else:
            out[bench] = {"groups": gsum, "macro": _macro(all_units)}
        missing = sorted(set(EXPECTED_GROUPS.get(bench, ())) - set(gsum))
        out[bench]["missing_groups"] = missing
        out[bench]["partial"] = bool(missing) or any(g["partial"] for g in gsum.values())
        out[bench]["n_episodes"] = sum(g["n_episodes"] for g in gsum.values())
        # A missing group counts when its size is known (LIBERO, LIBERO-Plus).
        unit_target = max(u["target"] for u in all_units)
        sizes = GROUP_SIZES.get(bench, {})
        out[bench]["target_episodes"] = sum(g["target_episodes"] for g in gsum.values()) + sum(
            sizes.get(g, 0) * unit_target for g in missing
        )
    return out


def print_summary(summary: dict[str, dict]) -> None:
    """Print a ``summarize`` result, one block per benchmark."""
    for bench, s in summary.items():
        flag = "  (partial)" if s["partial"] else ""
        print(f"\n== {bench}: macro {100 * s['macro']:.1f}%{flag}")
        if s["missing_groups"]:
            print(f"  missing: {', '.join(s['missing_groups'])}")
        for group, g in s["groups"].items():
            flag = "  (partial)" if g["partial"] else ""
            print(
                f"  {group:<48} pooled {100 * g['pooled']:5.1f}%  macro {100 * g['macro']:5.1f}%"
                f"  units {g['n_units']:>3}  eps {g['n_episodes']:>5}{flag}"
            )
            if g["missing_units"]:
                print(f"    missing units: {', '.join(g['missing_units'])}")
        if "per_axis" in s:
            for axis, m in s["per_axis"].items():
                print(f"  axis {axis:<43} {100 * m:5.1f}%")


# Short column names of the LIBERO-Plus axes.
AXIS_SHORT = {
    "camera_viewpoints": "cam",
    "robot_initial_states": "robot",
    "language_instructions": "lang",
    "light_conditions": "light",
    "background_textures": "bckg",
    "sensor_noise": "noise",
    "objects_layout": "layout",
}


def _dim(text: str, partial: bool) -> str:
    """Rich markup: grey while episodes or units are missing."""
    return f"[dim]{text}[/dim]" if partial else text


def clean2rand_cells(out: Path, scene_dirs: tuple[str, ...]) -> dict[str, str]:
    """``trials`` and ``avg`` (bold) of the Clean2Rand number under ``out``: the mean of the
    task means of ``scene_dirs``. Grey while a scene is partial or has no results."""
    summary = summarize(collect(out, scene_dirs))
    if not summary:
        return {}
    done = sum(s["n_episodes"] for s in summary.values())
    targets = [s["target_episodes"] for s in summary.values()]
    target = sum(targets) + (len(scene_dirs) - len(summary)) * max(targets)
    partial = done < target or any(s["partial"] for s in summary.values())
    mean = statistics.fmean(s["macro"] for s in summary.values())
    return {
        "trials": _dim(f"{done}/{target}", partial),
        "avg": _dim(f"[bold]{100 * mean:.1f}[/bold]", partial),
    }


def result_cells(out: Path, bench_dirs: tuple[str, ...]) -> dict[str, str]:
    """``trials`` (episodes done / expected), ``steps`` (mean env steps of the successful
    episodes), then success rates (%) under ``out``: one cell per suite / axis / group and one per
    benchmark (its macro), last. Cells with missing episodes or units are grey (rich markup)."""
    table = collect(out, bench_dirs)
    summary = summarize(table)
    cells: dict[str, str] = {}
    done = target = 0
    steps: list[int] = []
    for bench in bench_dirs:
        if bench not in summary:
            continue
        b = summary[bench]
        groups = b["groups"]
        # Column order: AXIS_SHORT, then EXPECTED_GROUPS.
        parts: dict[str, tuple[float, bool]] = {}
        if "per_axis" in b:
            per_axis = b["per_axis"]
            for axis, short in AXIS_SHORT.items():
                if axis in per_axis:
                    axis_partial = any(
                        v["partial"] for g, v in groups.items() if g.endswith(axis)
                    ) or any(g.endswith(axis) for g in b["missing_groups"])
                    parts[short] = (per_axis[axis], axis_partial)
        elif len(groups) > 1:
            order = {g: i for i, g in enumerate(EXPECTED_GROUPS.get(bench, ()))}
            for g in sorted(groups, key=lambda g: (order.get(g, len(order)), g)):
                parts[g.removeprefix("libero_")] = (groups[g]["pooled"], groups[g]["partial"])
        for name, (rate, partial) in parts.items():
            cells[name] = _dim(f"{100 * rate:.1f}", partial)
        cells[bench] = _dim(f"{100 * b['macro']:.1f}", b["partial"])
        done += b["n_episodes"]
        target += b["target_episodes"]
        steps += [s for units in table[bench].values() for u in units for s in u["success_steps"]]
    if not cells:
        return cells
    partial = done < target
    return {
        "trials": _dim(f"{done}/{target}", partial),
        "steps": _dim(f"{statistics.fmean(steps):.0f}", partial) if steps else "-",
        **cells,
    }
