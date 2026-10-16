"""Roll out a REKI checkpoint in a simulation benchmark (the CLI of ``reki.eval.rollout``).

    pixi run -e sim python scripts/eval/rollout.py --ckpt_dir=<ckpt> --task libero_spatial \\
        --log_dir=<out_dir>/libero

``launchers/eval_protocols.py`` prints the exact command of each released result; the env vars
and sim servers are in ``docs/evaluation.md``. LIBERO and LIBERO-Plus run in-process; RoboTwin
and RoboCasa365 drive a sim server over a socket (``--sim_host``/``--sim_port``, or
``--sim_server_cmd`` to start one per worker).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

from reki.eval.protocols import LIBERO_PLUS_AXES, N_EPISODES, libero_plus_category
from reki.eval.rollout.libero import LiberoAdapter
from reki.eval.rollout.loop import Adapter, RolloutArgs, run, warm_compile_cache
from reki.eval.rollout.remote import (
    RoboCasa365Adapter,
    RoboTwinAdapter,
    seed_cache_root,
)
from reki.eval.rollout.workers import (
    Claims,
    check_sharding,
    set_compile_cache_dir,
    spawn_workers,
)
from reki.inference.spec import INFERENCE_SPEC_FILE
from reki.policy.checkpoint import resolve_checkpoint
from reki.utils.cli import str2bool


def parser() -> argparse.ArgumentParser:
    """The command-line argument parser for simulation rollouts."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    def flag(name: str, default: bool, help: str) -> None:
        p.add_argument(name, type=str2bool, nargs="?", const=True, default=default, help=help)

    p.add_argument(
        "--ckpt_dir",
        required=True,
        help="a checkpoint directory, or an HF model id (e.g. kyutai/reki-2b-libero)",
    )
    p.add_argument(
        "--env_type",
        choices=("libero", "robotwin", "robocasa365"),
        default=None,
        help="libero runs in-process (also LIBERO-Plus); the others drive a sim server. "
        "Default: the benchmark of an exported checkpoint, else libero.",
    )
    p.add_argument(
        "--task",
        nargs="+",
        default=None,
        help="LIBERO suite(s), or socket-bench task name(s); comma lists work too. "
        "Default: the full grid (the 4 LIBERO suites; every task of the benchmark).",
    )
    p.add_argument(
        "--task_id", type=int, nargs="+", default=None, help="LIBERO task ids. Default: 0-9."
    )
    p.add_argument(
        "--n_episodes",
        type=int,
        default=None,
        help="episodes per unit. Default: the protocol (reki.eval.protocols.N_EPISODES)",
    )
    p.add_argument(
        "--n_action_steps",
        type=int,
        default=None,
        help="actions executed per predict (capped at the chunk). Default: the history stride "
        "(1 s); a K>1 checkpoint forces the stride",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42, help="episode seeds are drawn per unit")
    p.add_argument("--max_steps", type=int, default=None, help="default: the task's horizon")
    p.add_argument(
        "--history_k",
        type=int,
        default=None,
        help="history window to evaluate with (register ring of K-1 frames). "
        "Default: the trained window.",
    )
    p.add_argument("--speed_bin", type=int, default=None, help="'speed: N' prompt line")
    p.add_argument(
        "--log_dir",
        required=True,
        help="results dir",
    )
    flag("--save_videos", False, "write videos/epNNN_<success|fail>.mp4 per episode")
    flag(
        "--save_traj",
        False,
        "write per-episode traj/epNNN.npz with predicted chunks, executed actions, states, and "
        "the sim state (LIBERO and RoboCasa365 only) for debugging and bit-level comparisons "
        "of two code versions",
    )
    flag(
        "--resume",
        False,
        "skip units whose results.json has n_episodes episodes; extend shorter ones (same seeds)",
    )
    flag("--use_ema", True, "overlay ema.safetensors onto the weights when the checkpoint has it")
    flag(
        "--compile",
        True,
        "compile the expert and replay the prefill as CUDA graphs (RekiModel.compile_inference): "
        "one ~15-30 s compile per process, then ~7x faster predicts; =false runs eager",
    )
    p.add_argument(
        "--compile_cache_dir",
        default=None,
        help="compile cache directory, shared by the workers and later evals "
        "(default: $TORCHINDUCTOR_CACHE_DIR, else ~/.cache/reki/inductor)",
    )
    p.add_argument(
        "--suite_alias",
        default=None,
        help="LIBERO: results dir of the suite instead of its name (may nest, e.g. "
        "libero_plus_object/background_textures)",
    )
    p.add_argument(
        "--perturbation_axis",
        default=None,
        choices=tuple(map(libero_plus_category, LIBERO_PLUS_AXES)),
        help="LIBERO-Plus: run the task ids of this category (overrides --task_id)",
    )
    p.add_argument(
        "--robotwin_seed_cache",
        default=os.environ.get("REKI_ROBOTWIN_SEED_CACHE") or "auto",
        help="RoboTwin: root of the accepted-seed cache shared by every eval on the same scene "
        "code. A cached episode seed resets in seconds on its recorded scene and instruction "
        "instead of re-running the expert seed search. auto: ~/.cache/reki/robotwin_seed_cache, "
        "for the tasks the sim server can reset trusted; none: search every reset. "
        "Default: $REKI_ROBOTWIN_SEED_CACHE, else auto",
    )
    p.add_argument("--sim_host", default="127.0.0.1")
    p.add_argument(
        "--sim_port",
        default="5798",
        help="sim server port; with --num_workers K, a comma list of K running servers",
    )
    p.add_argument(
        "--sim_server_cmd",
        default=None,
        help="shell command that starts one sim server with --port 0 --port-file {portfile}; "
        "one is started per worker",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="rollout processes on this GPU, pulling units as they free up (same results)",
    )
    p.add_argument("--_claim_dir", default=None, help=argparse.SUPPRESS)
    p.add_argument("--_warm_compile_cache", action="store_true", help=argparse.SUPPRESS)
    return p


def protocol_episodes(env_type: str, perturbation_axis: str | None) -> int:
    """Episodes per unit of the released protocol."""
    return N_EPISODES["libero_plus" if perturbation_axis else env_type]


def make_adapter(args: argparse.Namespace, sim_port: int) -> Adapter:
    """The benchmark adapter selected by the rollout environment type."""
    if args.env_type == "libero":
        return LiberoAdapter(args.task, args.task_id, args.perturbation_axis, args.suite_alias)
    if args.env_type == "robotwin":
        return RoboTwinAdapter(
            args.task,
            args.ckpt_dir,
            args.sim_host,
            sim_port,
            *seed_cache_root(args.robotwin_seed_cache),
        )
    return RoboCasa365Adapter(args.task, args.ckpt_dir, args.sim_host, sim_port)


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    args = parser().parse_args(argv)
    if args.task is not None:
        args.task = [t for v in args.task for t in v.split(",") if t]
    args.ckpt_dir = str(resolve_checkpoint(args.ckpt_dir, include_ema=True))
    if args.env_type is None:
        spec_p = Path(args.ckpt_dir) / INFERENCE_SPEC_FILE
        bench = json.loads(spec_p.read_text()).get("benchmark") if spec_p.exists() else None
        args.env_type = bench or "libero"
    if args.n_episodes is None:
        args.n_episodes = protocol_episodes(args.env_type, args.perturbation_axis)
    sim_ports = [int(x) for x in args.sim_port.split(",") if x]
    if not sim_ports:
        raise SystemExit(f"invalid --sim_port {args.sim_port!r}")
    if args.compile:
        print(f"[compile] compile cache {set_compile_cache_dir(args.compile_cache_dir)}")
    sharded = args.num_workers > 1 or args.sim_server_cmd is not None or len(sim_ports) > 1
    if args._claim_dir is None and sharded:
        check_sharding(args.env_type != "libero", args.num_workers, args.sim_server_cmd, sim_ports)
        units = [u.key for u in make_adapter(args, sim_ports[0]).units()]
        spawn_workers(
            units,
            args.num_workers,
            args.sim_server_cmd,
            remote_ports=sim_ports if len(sim_ports) > 1 else None,
            warm_compile=args.compile and torch.device(args.device).type == "cuda",
        )
        return
    rollout_args = RolloutArgs(
        ckpt_dir=args.ckpt_dir,
        n_episodes=args.n_episodes,
        n_action_steps=args.n_action_steps,
        device=args.device,
        seed=args.seed,
        max_steps=args.max_steps,
        history_k=args.history_k,
        log_dir=args.log_dir,
        save_videos=args.save_videos,
        save_traj=args.save_traj,
        resume=args.resume,
        speed_bin=args.speed_bin,
        use_ema=args.use_ema,
        compile=args.compile,
    )
    if args._warm_compile_cache:
        warm_compile_cache(make_adapter(args, sim_ports[0]), rollout_args)
        return
    run(make_adapter(args, sim_ports[0]), rollout_args, Claims(args._claim_dir))


if __name__ == "__main__":
    main()
