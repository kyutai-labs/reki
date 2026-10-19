"""The episode loop shared by every benchmark: policy setup, predict cadence, resume, videos and
the ``results.json`` of each unit. A benchmark plugs in through an ``Adapter``."""

from __future__ import annotations

import json
import os
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS
from PIL import Image

from reki.eval.results import read_leaf
from reki.eval.rollout.traj import RolloutTraj
from reki.eval.rollout.workers import Claims
from reki.inference.benchmarks import BENCHMARKS
from reki.inference.features import EnvObs, build_inference_sample
from reki.inference.policy import build_eval_preprocessor, build_policy
from reki.inference.spec import (
    INFERENCE_SPEC_FILE,
    SHARED_STATS_KEY,
    FeatStats,
    Stats,
    benchmark_stats,
    history_stream_n_action_steps,
    load_inference_spec,
    select_stats,
    stats_dim,
)
from reki.policy.checkpoint import load_pretrained_weights
from reki.policy.train_config import TrainConfig
from reki.vlm.qwen3_5.preprocessor import CAMERA_ROLE_TOKEN_IDS


@dataclass(frozen=True)
class Unit:
    """One cell of a benchmark's work grid: one ``results.json``."""

    key: str  # unique in the grid (worker claims)
    rel_dir: str  # results dir under the eval dir
    task: str  # simulator task (LIBERO: the suite); the stats key of per-task stats
    fields: dict[str, Any]  # identity fields of results.json


class Env(Protocol):
    """One task's simulator. ``step`` returns (raw obs, success, terminated, truncated)."""

    max_episode_steps: int

    def reset(self, seed: int, episode: int) -> dict: ...
    def step(self, action: np.ndarray, *, need_obs: bool, sim_state: bool) -> tuple: ...
    def observe(self, raw: dict) -> EnvObs: ...
    def frame(self, raw: dict) -> np.ndarray: ...
    def traj_start(self, raw: dict) -> dict: ...
    def traj_step(self, raw: dict) -> dict: ...


class Adapter:
    """The parts of a rollout that differ between benchmarks.

    ``benchmark`` is a key of ``BENCHMARKS``. ``tag`` prefixes the log lines. ``n_cameras`` and
    ``image_hw`` are the camera count and image size of the policy; the compile warm-up uses
    them."""

    benchmark: str
    tag = ""
    n_cameras = 1
    image_hw = (256, 256)
    per_task_stats = False
    skip_obs_within_chunk = False
    camera_indices: list[int] | None = None

    def configure(self, train_cfg: TrainConfig) -> None:
        """Read what the adapter needs from the checkpoint's config."""

    def prompt_fps(self, stats_fps: float) -> float:
        """The fps of the prompt and of ``chunk_duration_s``."""
        return 20

    def env_fps(self, stats_fps: float) -> int:
        """The sim control rate: of ``history_stride_s`` and of the videos."""
        return 20

    def units(self) -> list[Unit]:
        raise NotImplementedError

    def session(self) -> AbstractContextManager:
        raise NotImplementedError

    def make_env(self, unit: Unit) -> Env:
        raise NotImplementedError

    def decode(self, action_norm: torch.Tensor, stats: FeatStats) -> np.ndarray:
        raise NotImplementedError


@dataclass
class RolloutArgs:
    """Configuration for one evaluation rollout process."""

    ckpt_dir: str
    n_episodes: int
    log_dir: str
    n_action_steps: int | None = None
    device: str = "cuda"
    seed: int = 42
    max_steps: int | None = None
    history_k: int | None = None
    save_videos: bool = False
    save_traj: bool = False
    resume: bool = False
    speed_bin: int | None = None
    use_ema: bool = True
    compile: bool = True


def eval_stats(ckpt_path: Path, benchmark: str, train_cfg: TrainConfig) -> tuple[float, Stats]:
    """(prompt fps, stats): from the checkpoint when it ships them
    (``scripts/export_checkpoint.py --benchmark``), else from the training dataset under
    ``REKI_DATASETS_ROOT`` (a raw training checkpoint)."""
    if (ckpt_path / INFERENCE_SPEC_FILE).exists():
        spec, stats = load_inference_spec(ckpt_path)
        if spec.benchmark != benchmark:
            raise ValueError(f"{ckpt_path} is exported for {spec.benchmark!r}, not {benchmark!r}")
        return spec.prompt_fps, stats
    return benchmark_stats(benchmark, train_cfg)


def _task_stats(stats: Stats, task: str | None) -> tuple[FeatStats, FeatStats]:
    """The stats of ``task``: the shared set, or the task's own set when they are per task."""
    per_task = SHARED_STATS_KEY not in stats
    return select_stats(stats, per_task, task if per_task else None)


def override_history_k(train_cfg: TrainConfig, history_k: int | None, tag: str) -> int:
    """Set the eval history window on ``train_cfg`` before the policy is built. The register
    ring holds K-1 frames. Return the trained K, which sets the predict cadence. A window longer
    than the trained one is allowed and prints a message."""
    trained_k = train_cfg.history_k
    if history_k is not None and history_k != trained_k:
        longer = " (longer than trained: extrapolating)" if history_k > trained_k else ""
        print(
            f"[{tag}] history window {trained_k} -> {history_k} frames (register capacity "
            f"{history_k - 1}){longer}; the predict cadence stays at the trained stride."
        )
        train_cfg.history_k = history_k
    return trained_k


def _prompt_debug_lines(unit: str, batch: dict[str, Any], preprocessor: Any, obs: EnvObs) -> str:
    """The conditioning the model sees at the first step: prompt, camera-role tokens, image."""
    im = np.asarray(obs.images[-1], np.float32)
    ids = batch[OBS_LANGUAGE_TOKENS].detach().cpu().tolist()
    img_pad = getattr(preprocessor, "image_pad_id", None)
    txt = preprocessor.tokenizer.decode([t for t in ids if t != img_pad], skip_special_tokens=False)
    role = [t for t in ids if t in CAMERA_ROLE_TOKEN_IDS.values()]
    return (
        f"[dbg {unit}] PROMPT: {txt!r}\n"
        f"[dbg {unit}] n_cams={len(obs.images)} camera_role_ids={role} "
        f"n_image_pad={sum(t == img_pad for t in ids)}\n"
        f"[dbg {unit}] image shape={im.shape} mean={im.mean():.1f} std={im.std():.1f} "
        f"min={im.min():.0f} max={im.max():.0f}"
    )


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    import imageio

    with imageio.get_writer(str(path), fps=fps) as writer:
        for f in frames:
            writer.append_data(f)  # type: ignore[attr-defined]


def _complete(results_file: Path, n_episodes: int) -> bool:
    """True if ``results_file`` holds at least ``n_episodes`` episodes."""
    try:
        return len(json.loads(results_file.read_text()).get("episodes", [])) >= n_episodes
    except (OSError, json.JSONDecodeError):
        return False


def _prior_episodes(results_file: Path, run_meta: dict[str, Any]) -> list[dict[str, Any]]:
    """``--resume``: the episodes already in ``results_file``. The run then adds episodes up to
    the new count (e.g. 10 -> 20). Each ``run_meta`` key in the file must match the run config.
    The caller checks each episode seed against its own draw."""
    if not results_file.exists():
        return []
    prior = json.loads(results_file.read_text())

    def same(k: str, a: Any, b: Any) -> bool:
        return Path(a).parts[-2:] == Path(b).parts[-2:] if k == "ckpt_dir" else a == b

    mismatched = {
        k: (prior[k], v) for k, v in run_meta.items() if k in prior and not same(k, prior[k], v)
    }
    if mismatched:
        raise ValueError(f"--resume: {results_file} has another config (old, new): {mismatched}")
    if unchecked := sorted(k for k in run_meta if k not in prior):
        print(f"  [resume] {results_file} has no {unchecked}; not verified")
    return prior["episodes"]


class _Rollout:
    """The policy and settings of one run, shared by its units."""

    def __init__(self, adapter: Adapter, a: RolloutArgs) -> None:
        self.adapter, self.a = adapter, a
        ckpt_path = Path(a.ckpt_dir).resolve()
        cfg = TrainConfig.from_checkpoint(ckpt_path)
        self.train_cfg = cfg
        trained_k = override_history_k(cfg, a.history_k, adapter.tag)
        stats_fps, self.stats = eval_stats(ckpt_path, adapter.benchmark, cfg)
        adapter.configure(cfg)

        self.policy = build_policy(cfg, a.device).to(a.device)
        # An export has the EMA merged and no ema.safetensors.
        use_ema = a.use_ema and (ckpt_path / "ema.safetensors").exists()
        load_pretrained_weights(self.policy, str(ckpt_path), use_ema=use_ema)
        self.policy.eval()
        if a.compile and torch.device(a.device).type == "cuda":
            self.policy.compile_inference()
            # A silent eager fallback would run part of the eval with other numerics (and far
            # slower): raise instead.
            torch._dynamo.config.fail_on_recompile_limit_hit = True
        self.preprocessor = build_eval_preprocessor(cfg)

        self.prompt_fps = adapter.prompt_fps(stats_fps)
        self.env_fps = adapter.env_fps(stats_fps)
        stride_steps = max(1, round(cfg.history_stride_s * self.env_fps))
        # Default: one predict per history stride (1 s), at any K, as RekiPolicy.
        n_action_steps = stride_steps if a.n_action_steps is None else a.n_action_steps
        if cfg.chunk_duration_s is not None:
            chunk_size = max(1, round(cfg.chunk_duration_s * self.prompt_fps))
            self.policy.config.chunk_size = chunk_size
            n_action_steps = min(n_action_steps, chunk_size)
        self.n_action_steps = history_stream_n_action_steps(
            cfg, trained_k, n_action_steps, stride_steps, self.policy.config.chunk_size, adapter.tag
        )
        self.skip_obs = adapter.skip_obs_within_chunk and not a.save_videos
        self.run_meta = {
            "ckpt_dir": str(ckpt_path),
            "n_action_steps": self.n_action_steps,
            "trained_history_k": trained_k,
            "eval_history_k": cfg.history_k,
            "history_stride_s": cfg.history_stride_s,
            "action_expert_num_registers": cfg.action_expert_num_registers,
            "fps": self.env_fps,
            "seed": a.seed,
            "use_ema": use_ema,
        }
        print(
            f"[{adapter.tag}] policy ready on {a.device}: chunk_size={self.policy.config.chunk_size}"
            f", n_action_steps={self.n_action_steps}, history_k={cfg.history_k}, "
            f"fps={self.env_fps}, skip_obs={self.skip_obs}"
        )

    def run_unit(self, unit: Unit, log_path: Path) -> None:
        a = self.a
        state_stats, action_stats = _task_stats(
            self.stats, unit.task if self.adapter.per_task_stats else None
        )
        env = self.adapter.make_env(unit)
        cap = a.max_steps if a.max_steps is not None else env.max_episode_steps
        log_path.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {unit.key} -> {log_path} (cap={cap}) ===")

        episodes = _prior_episodes(log_path / "results.json", self.run_meta) if a.resume else []
        n_prior = len(episodes)
        if n_prior:
            print(f"  [resume] {unit.key}: keeping {n_prior} episodes")
        rng = np.random.default_rng(a.seed)
        for ep in range(a.n_episodes):
            ep_seed = int(rng.integers(0, 2**31 - 1))
            if ep < n_prior:
                if episodes[ep]["seed"] != ep_seed:
                    raise ValueError(
                        f"--resume: {unit.key} ep{ep} ran with seed {episodes[ep]['seed']}, "
                        f"--seed={a.seed} draws {ep_seed}"
                    )
                continue
            episodes.append(
                self._episode(env, unit.key, ep, ep_seed, cap, state_stats, action_stats, log_path)
            )
            n_success = sum(e["success"] for e in episodes)
            results = {
                **unit.fields,
                "n_episodes": len(episodes),
                "n_success": n_success,
                "success_rate": n_success / len(episodes),
                "target_episodes": a.n_episodes,
                **self.run_meta,
                "episodes": episodes,
            }
            (log_path / "results.json").write_text(json.dumps(results, indent=2))
        n_success = sum(e["success"] for e in episodes)
        pct = 100 * n_success / a.n_episodes if a.n_episodes else 0
        print(f"=== {unit.key}: {n_success}/{a.n_episodes} ({pct:.1f}%) ===")

    def _batch(self, obs: EnvObs, state_stats: FeatStats, action_dim: int) -> dict[str, Any]:
        bench = BENCHMARKS[self.adapter.benchmark]
        return build_inference_sample(
            obs,
            preprocessor=self.preprocessor,
            train_cfg=self.train_cfg,
            state_stats=state_stats,
            embodiment=bench.embodiment,
            control_mode=bench.control_mode,
            device=self.a.device,
            action_dim=action_dim,
            dataset=bench.dataset,
            camera_indices=self.adapter.camera_indices,
            speed_bin=self.a.speed_bin,
            robot_type=bench.robot_type,
            fps=self.prompt_fps,
        )

    def warm_up(self) -> None:
        """Run predicts on random observations of the benchmark's shape, past the register
        cache capacity. This compiles every graph into the compile cache. The workers that
        start next load the graphs from it. No env or sim server is used."""
        unit = self.adapter.units()[0]
        state_stats, action_stats = _task_stats(
            self.stats, unit.task if self.adapter.per_task_stats else None
        )
        rng = np.random.default_rng(0)
        h, w = self.adapter.image_hw
        t0 = time.perf_counter()
        self.policy.reset_episode()
        for _ in range(max(2, self.train_cfg.history_k + 1)):
            obs = EnvObs(
                images=[
                    Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
                    for _ in range(self.adapter.n_cameras)
                ],
                state_raw=rng.uniform(-1, 1, stats_dim(state_stats)).astype(np.float32),
                task_text="pick up the black bowl and place it on the plate",
            )
            with torch.no_grad():
                self.policy.predict_action_chunk(
                    self._batch(obs, state_stats, stats_dim(action_stats))
                )
        torch.cuda.synchronize()
        print(
            f"[compile] warm-up done in {time.perf_counter() - t0:.1f} s "
            f"({_compile_seconds():.1f} s compiling), cache {os.environ.get('TORCHINDUCTOR_CACHE_DIR')}"
        )

    def _episode(
        self,
        env: Env,
        unit: str,
        ep: int,
        seed: int,
        cap: int,
        state_stats: FeatStats,
        action_stats: FeatStats,
        log_path: Path,
    ) -> dict[str, Any]:
        a, adapter, policy = self.a, self.adapter, self.policy
        true_dim = stats_dim(action_stats)
        torch.manual_seed(seed)
        raw = env.reset(seed, episode=ep)
        policy.reset_episode()
        traj = RolloutTraj(env.traj_start(raw)) if a.save_traj else None
        obs = env.observe(raw)
        frames = [env.frame(raw)] if a.save_videos else []
        steps, success, terminal = 0, False, "cap"
        while steps < cap:
            batch = self._batch(obs, state_stats, true_dim)
            if ep == 0 and steps == 0:
                print(_prompt_debug_lines(unit, batch, self.preprocessor, obs))
            # Memory frames this predict reads: the cache outlives reset_episode.
            ring = policy._register_cache
            ring_depth = 0 if ring is None else ring.n_frames
            with torch.no_grad():
                chunk = policy.predict_action_chunk(batch)
            if ep == 0 and steps == 0 and a.compile:
                print(f"[compile] first predict: {_compile_seconds():.1f} s compiling")
            if traj is not None:
                traj.predict(
                    step=steps,
                    state=obs.state_raw,
                    chunk=chunk[0, :, :true_dim],
                    ring_depth=ring_depth,
                )
            chunk = chunk[0, : self.n_action_steps, :true_dim].to(torch.float32)
            for t in range(chunk.shape[0]):
                action = adapter.decode(chunk[t], action_stats)
                need_obs = not self.skip_obs or t == chunk.shape[0] - 1
                raw, ok, terminated, truncated = env.step(
                    action.astype(np.float32), need_obs=need_obs, sim_state=traj is not None
                )
                steps += 1
                if traj is not None:
                    traj.step(action, env.traj_step(raw))
                if need_obs:
                    obs = env.observe(raw)
                if a.save_videos:
                    frames.append(env.frame(raw))
                if ok:
                    success, terminal = True, "success"
                    break
                if terminated or truncated:
                    terminal = "step_lim" if truncated else "terminated"
                    break
            if success or steps >= cap:
                break
        print(f"  ep {ep + 1:2d}: steps={steps:3d} success={success}")
        if traj is not None:
            (log_path / "traj").mkdir(exist_ok=True)
            traj.save(
                log_path / "traj" / f"ep{ep:03d}.npz",
                episode=ep,
                seed=seed,
                success=success,
                terminal=terminal,
                steps=steps,
                fps=self.env_fps,
                n_action_steps=self.n_action_steps,
                history_k=self.train_cfg.history_k,
            )
        if frames:
            (log_path / "videos").mkdir(exist_ok=True)
            tag = "success" if success else "fail"
            _write_video(log_path / "videos" / f"ep{ep:03d}_{tag}.mp4", frames, self.env_fps)
        return {
            "episode": ep,
            "seed": seed,
            "steps": steps,
            "success": success,
            "terminal": terminal,
        }


def _compile_seconds() -> float:
    """Seconds this process spent in torch.compile (tracing and code generation)."""
    return torch._dynamo.utils.calculate_time_spent().get("entire_frame_compile", 0.0)


def warm_compile_cache(adapter: Adapter, a: RolloutArgs) -> None:
    """Build the policy as a worker does and fill the compile cache (``_Rollout.warm_up``)."""
    _Rollout(adapter, a).warm_up()


def run(adapter: Adapter, a: RolloutArgs, claims: Claims | None = None) -> None:
    """Run every unit of ``adapter`` this worker claims; with ``resume``, skip finished units
    and extend shorter ones."""
    claims = claims or Claims(None)
    rollout = _Rollout(adapter, a)
    base = Path(a.log_dir).resolve()
    units = adapter.units()
    if a.resume:
        done = [u for u in units if _complete(base / u.rel_dir / "results.json", a.n_episodes)]
        for u in done:
            print(f"  skip {u.key} (done: {base / u.rel_dir / 'results.json'})")
        units = [u for u in units if u not in done]
    if units:
        with adapter.session():
            for unit in units:
                if claims.take(unit.key):
                    rollout.run_unit(unit, base / unit.rel_dir)
    _print_totals(base)


def _print_totals(base: Path) -> None:
    runs = [r for p in sorted(base.rglob("results.json")) if (r := read_leaf(p)) is not None]
    ns, ne = sum(r["n_success"] for r in runs), sum(r["n_episodes"] for r in runs)
    rate = f"{100 * ns / ne:.1f}%" if ne else "-"
    print(f"\n=== {base}: {ns}/{ne} = {rate} over {len(runs)} units on disk ===")
