"""
Standalone training script for the robotics VLA (plain Accelerate loop).

Usage:
    pixi run train --config=configs/sft.yaml --pretrained_path=kyutai/reki-2b --dataset=libero \\
        --output_dir=/path/to/run

CLI flags override the yaml (``--steps=100``); without ``--config``, every field
comes from the flags. ``output_dir`` is required either way.

Resume from checkpoint:
    pixi run train --config=configs/sft.yaml --output_dir=/path/to/existing/run --resume=True
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import random
import subprocess
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Any

import fire
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, GradientAccumulationPlugin, set_seed
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import ConcatDataset, DataLoader
from transformers import get_cosine_with_min_lr_schedule_with_warmup, get_wsd_schedule

from reki.data.robotics.collate import combined_collate
from reki.data.robotics.combined_dataset import CombinedLeRobotDataset
from reki.data.robotics.factory import make_combined_dataset
from reki.policy.checkpoint import load_checkpoint_tensors, load_pretrained_weights
from reki.policy.configuration import RekiConfig
from reki.policy.constants import ACTION
from reki.policy.external import get_external_policy
from reki.policy.external.base import ExternalPolicy
from reki.policy.modeling import RekiModel
from reki.policy.optim import EMAModel, StochasticRoundingAdamW, build_optimizer_groups
from reki.policy.train_config import TrainConfig
from reki.policy.val import prefetch_val_batches, val_flow_loss
from reki.utils.cli import fire_bool
from reki.utils.dist_utils import is_rank_zero

warnings.filterwarnings("ignore", "Cannot enable progress bars", UserWarning, "datasets")
torch.multiprocessing.set_sharing_strategy("file_system")

ZERO1_OPTIMIZER = "zero1_optimizer.pt"


def build_accelerator(cfg: TrainConfig) -> Accelerator:
    """Build the configured Accelerate runtime."""
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=cfg.ddp_find_unused_parameters,
        static_graph=cfg.ddp_static_graph,
        bucket_cap_mb=cfg.ddp_bucket_cap_mb,
        gradient_as_bucket_view=True,
    )
    return Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=cfg.gradient_accumulation_steps, sync_with_dataloader=False
        ),
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs],
    )


def configure_torch(seed: int) -> None:
    """Seed the process and configure the training CUDA backends."""
    set_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True


def build_dataloader(cfg: TrainConfig) -> tuple[CombinedLeRobotDataset | ConcatDataset, DataLoader]:
    """Build the training dataset and data loader."""
    # The builders check the rank without an Accelerator, so dry_run.py can call them.
    if is_rank_zero():
        print("Loading dataset")
    ds, sampler = make_combined_dataset(cfg)
    collate_fn = (
        get_external_policy(cfg.external_policy).collate
        if cfg.external_policy is not None
        else partial(
            combined_collate,
            max_image_tokens=cfg.max_image_tokens,
            max_text_tokens=cfg.max_text_tokens,
        )
    )
    dataloader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
    )
    return ds, dataloader


def build_policy(
    cfg: TrainConfig,
    ds: CombinedLeRobotDataset | ConcatDataset,
    device: str = "cuda",
    dtype: str = "bfloat16",
) -> RekiModel | ExternalPolicy:
    """Build and configure the selected policy. ``device`` and ``dtype`` apply to RekiModel only;
    an external policy is moved by ``accelerator.prepare`` (or the caller)."""
    if is_rank_zero():
        print("Building model")
    if cfg.external_policy is not None:
        adapter = get_external_policy(cfg.external_policy)
        assert isinstance(ds, CombinedLeRobotDataset)  # external policies train without VL
        policy = adapter.build(cfg, ds.external_policy_stats(), ds.external_chunk_size())
        policy.freeze_vision_blocks(cfg.train_vision_blocks)
        if cfg.gradient_checkpointing:
            policy.enable_gradient_checkpointing()
    else:
        policy = RekiModel(RekiConfig.from_train_config(cfg, device=device, dtype=dtype))
        if cfg.pretrained_path:
            load_pretrained_weights(policy, cfg.pretrained_path)
        if cfg.gradient_checkpointing:
            policy.model.vlm_with_expert.gradient_checkpointing = True
    num_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in policy.parameters())
    if is_rank_zero():
        print(f"Parameters: {num_trainable:,} trainable / {num_total:,} total")
    return policy


def build_optimizer(
    cfg: TrainConfig, policy: RekiModel | ExternalPolicy, zero1: bool = False
) -> tuple[torch.optim.Optimizer, LambdaLR]:
    """Build the optimizer parameter groups and learning-rate scheduler.

    ``zero1`` shards the optimizer state over the ranks (ZeroRedundancyOptimizer). Its flat
    buckets need one dtype, so the fp32 params are cast to bf16.
    """
    if isinstance(policy, ExternalPolicy):
        param_groups = policy.param_groups(
            lr=cfg.lr, weight_decay=cfg.weight_decay, backbone_lr_scale=cfg.backbone_lr_scale
        )
        group_info = {
            f"external:{i}@lr={g['lr']:.2e}": sum(p.numel() for p in g["params"])
            for i, g in enumerate(param_groups)
        }
    else:
        param_groups, group_info = build_optimizer_groups(
            policy,
            policy.model.vlm_with_expert,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            backbone_lr_scale=cfg.backbone_lr_scale,
        )
    if is_rank_zero():
        for name, count in group_info.items():
            print(f"  Optimizer group [{name}]: {count:,} params")
    if zero1:
        if cfg.gradient_accumulation_steps != 1:
            raise ValueError("use_zero1 requires gradient_accumulation_steps == 1")
        fp32_cast = 0
        for group in param_groups:
            for p in group["params"]:
                if p.dtype == torch.float32:
                    p.data = p.data.bfloat16()
                    fp32_cast += 1
        optimizer = ZeroRedundancyOptimizer(
            param_groups,
            optimizer_class=StochasticRoundingAdamW,
            betas=(0.9, 0.95),
            master_weights=cfg.optimizer_master_weights,
        )
        if is_rank_zero():
            print(f"  ZeRO-1 optimizer ({fp32_cast} fp32 params cast to bf16)")
    else:
        optimizer = StochasticRoundingAdamW(
            param_groups, betas=(0.9, 0.95), master_weights=cfg.optimizer_master_weights
        )
    if cfg.lr_schedule == "cosine":
        lr_scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.warmup_steps,
            num_training_steps=cfg.steps,
            min_lr_rate=cfg.min_lr / cfg.lr if cfg.lr > 0 else 0.0,
        )
    elif cfg.lr_schedule == "wsd":
        lr_scheduler = get_wsd_schedule(
            optimizer,
            num_warmup_steps=cfg.warmup_steps,
            num_decay_steps=cfg.decay_steps,
            num_training_steps=cfg.steps,
            min_lr_ratio=cfg.min_lr / cfg.lr if cfg.lr > 0 else 0.0,
        )
    else:
        raise ValueError(f"Unknown lr_schedule: {cfg.lr_schedule!r}")
    return optimizer, lr_scheduler


def _micro_step(
    accelerator: Accelerator,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Any],
    grad_clip: float,
) -> tuple[torch.Tensor, dict[str, Any], Any]:
    """Forward and backward one micro-batch under ``accelerator.accumulate``.

    Only the last micro-batch of an accumulation group clips and steps: before it, the prepared
    optimizer's ``step``/``zero_grad`` are no-ops and ``grad_norm`` is None.
    """
    grad_norm = None
    with accelerator.accumulate(policy):
        with accelerator.autocast():
            loss, loss_dict = policy.forward(batch)
        accelerator.backward(loss)
        if accelerator.sync_gradients and grad_clip > 0:
            grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip)
        optimizer.step()
        # Zeroed in place with accumulation: freed grads would make each no_sync backward allocate a
        # second copy next to the DDP buckets.
        optimizer.zero_grad(set_to_none=accelerator.gradient_accumulation_steps == 1)
    return loss, loss_dict, grad_norm


def _finish_optimizer_step(
    accelerator: Accelerator,
    policy: torch.nn.Module,
    lr_scheduler: Any,
    emas: dict[str, EMAModel],
    step: int,
) -> None:
    """Advance the LR schedule and EMA shadows once per optimizer step."""
    lr_scheduler.step()
    if emas:
        unwrapped = accelerator.unwrap_model(policy)
        for ema in emas.values():
            ema.update(unwrapped, step=step)


def _git_commit() -> str | None:
    """HEAD of the checkout that runs training; None outside a git checkout."""
    repo = Path(__file__).resolve().parents[1]
    if not (repo / ".git").exists():
        return None
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _batch_token_counts(batch: dict[str, Any], external: bool) -> dict[str, int]:
    """Samples and tokens of one micro-batch."""
    if external:
        return {
            "n_samples": int(batch[ACTION].shape[0]),
            "n_action_tokens": int(batch[ACTION].shape[0] * batch[ACTION].shape[1]),
        }
    counts = {
        "n_samples": int(batch["seq_lens"].shape[0]),
        "n_text_tokens": int(batch["observation.language.tokens"].shape[0]),
        "max_sample_tokens": int(batch["seq_lens"].max()),
    }
    n_img = 0
    for vi in batch.get("vision_inputs", []) or []:
        grid = vi.get("image_grid_thw") if isinstance(vi, dict) else None
        if grid is not None:
            n_img += int(grid.prod(dim=-1).sum().item()) // 4
    if n_img:
        counts["n_image_tokens"] = n_img
    pads = batch.get("action_is_pad")
    if isinstance(pads, list):
        n_act = sum(int((~p).sum()) for p in pads if p is not None)
        if n_act:
            counts["n_action_tokens"] = n_act
    return counts


class MetricLogger:
    """Train and val metrics to stdout and ``<output_dir>/metrics.jsonl`` (main rank).

    The train loss is the mean over all ranks of the micro-batch losses since the last log.
    Each of ``SUB_LOSSES`` is the mean over the micro-batches that have it.
    """

    SUB_LOSSES = ("flow_loss", "ce_loss", "subtask_loss", "text_loss")

    def __init__(self, cfg: TrainConfig, accelerator: Accelerator) -> None:
        self.accelerator = accelerator
        self.external = cfg.external_policy is not None
        self.jsonl_path = os.path.join(cfg.output_dir, "metrics.jsonl")
        self.loss_sum = torch.zeros((), device=accelerator.device)
        self.n_micro = 0
        self.sub_sum = torch.zeros(len(self.SUB_LOSSES), device=accelerator.device)
        self.sub_count = torch.zeros(len(self.SUB_LOSSES), device=accelerator.device)
        self.step_s: list[float] = []
        self.data_s: list[float] = []
        self.windows: list[float] = []

    def add_micro_batch(self, loss: torch.Tensor, loss_dict: dict[str, Any]) -> None:
        self.loss_sum += loss.detach()
        self.n_micro += 1
        for i, key in enumerate(self.SUB_LOSSES):
            if key in loss_dict:
                self.sub_sum[i] += loss_dict[key]
                self.sub_count[i] += 1

    def add_step(self, step_s: float, data_s: float, windows: float) -> None:
        self.step_s.append(step_s)
        self.data_s.append(data_s)
        self.windows.append(windows)

    def log_train(
        self, step: int, lr: float, grad_norm: Any, batch: dict[str, Any], gate_stats: dict
    ) -> None:
        """Collect and log train metrics; every rank must call this collective.

        The token counts are those of the main rank's last micro-batch.
        """
        loss_pack = torch.cat(
            [
                self.loss_sum.view(1),
                torch.tensor([float(self.n_micro)], device=self.accelerator.device),
                self.sub_sum,
                self.sub_count,
            ]
        )
        loss_reduced = self.accelerator.gather(loss_pack.unsqueeze(0)).sum(0)
        self.loss_sum.zero_()
        self.n_micro = 0
        self.sub_sum.zero_()
        self.sub_count.zero_()
        if not self.accelerator.is_main_process:
            self.step_s.clear()
            self.data_s.clear()
            self.windows.clear()
            return
        g_loss = (loss_reduced[0] / loss_reduced[1].clamp_min(1.0)).item()
        n = len(self.step_s)
        metrics: dict[str, Any] = {
            "step": step,
            "loss": round(g_loss, 5),
            "lr": float(lr),
            "windows": round(sum(self.windows) / n, 2),
            "step_s": round(sum(self.step_s) / n, 2),
            "data_s": round(sum(self.data_s) / n, 2),
        }
        n_sub = len(self.SUB_LOSSES)
        sub_sum, sub_count = loss_reduced[2 : 2 + n_sub], loss_reduced[2 + n_sub :]
        for key, total, count in zip(self.SUB_LOSSES, sub_sum.tolist(), sub_count.tolist()):
            if count > 0:
                metrics[key] = round(total / count, 5)
        if grad_norm is not None:
            gn = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            metrics["grad_norm"] = round(gn, 3)
        metrics.update(_batch_token_counts(batch, self.external))
        metrics.update(gate_stats)
        if torch.cuda.is_available():
            metrics["mem_peak_alloc_gb"] = round(torch.cuda.max_memory_allocated(0) / 1024**3, 2)
            metrics["mem_peak_reserved_gb"] = round(torch.cuda.max_memory_reserved(0) / 1024**3, 2)
            torch.cuda.reset_peak_memory_stats(0)
        self.step_s.clear()
        self.data_s.clear()
        self.windows.clear()
        self.log(metrics)
        torch.cuda.empty_cache()

    def log(self, metrics: dict[str, Any]) -> None:
        if not self.accelerator.is_main_process:
            return
        stdout_metrics = {**metrics}
        if "lr" in stdout_metrics:
            stdout_metrics["lr"] = f"{stdout_metrics['lr']:.2e}"
        print(stdout_metrics)
        with open(self.jsonl_path, "a") as f:
            json.dump(metrics, f)
            f.write("\n")


def _validate(
    cfg: TrainConfig,
    accelerator: Accelerator,
    policy: torch.nn.Module,
    val_batches: Any,
    emas: dict[str, EMAModel],
) -> dict[str, Any]:
    """Evaluate live or EMA policy variants and return their validation metrics."""
    variants = [(f"_{stem}", ema) for stem, ema in emas.items()] if emas else [("", None)]
    metrics: dict[str, Any] = {}
    for suffix, ema in variants:
        ctx = (
            ema.applied(accelerator.unwrap_model(policy))
            if ema is not None
            else contextlib.nullcontext()
        )
        with ctx:
            flow_loss = val_flow_loss(
                policy,
                val_batches,
                accelerator.device,
                seed=cfg.val_seed,
                accelerator=accelerator,
            )
        metrics[f"val/{cfg.val_dataset}/flow_loss{suffix}"] = round(flow_loss.item(), 5)
    return metrics


def _find_latest_checkpoint(output_dir: str) -> str | None:
    """Find the latest checkpoint-N directory in output_dir."""
    ckpt_dirs = sorted(
        Path(output_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])
    )
    return str(ckpt_dirs[-1]) if ckpt_dirs else None


def _prune_stale_optimizer_state(
    output_dir: str, keep_step: int, also_keep: int | None = None
) -> None:
    """Drop optimizer state from every checkpoint before ``keep_step``, except ``also_keep``.

    Resume uses the latest checkpoint (``_find_latest_checkpoint``), so older ones need only
    weights: ``_load_checkpoint`` loads weights only when there is no optimizer file. Saves
    ~26 GB per checkpoint. ``also_keep`` is the WSD decay start, so a longer run or another
    decay length can resume from it.
    """
    keep_dir = Path(output_dir) / f"checkpoint-{keep_step}"
    if not any(keep_dir.glob("optimizer*.bin")) and not (keep_dir / ZERO1_OPTIMIZER).exists():
        raise RuntimeError(f"refusing to prune: no optimizer state written in {keep_dir}")
    for ckpt in Path(output_dir).glob("checkpoint-*"):
        ckpt_step = int(ckpt.name.split("-")[1])
        if ckpt_step >= keep_step or ckpt_step == also_keep:
            continue
        for stale in (*ckpt.glob("optimizer*.bin"), *ckpt.glob(ZERO1_OPTIMIZER)):
            stale.unlink()
            print(f"Pruned stale optimizer state {stale}")


def _consolidate_zero1(cfg: TrainConfig, optimizer: torch.optim.Optimizer) -> None:
    """Gather the ZeRO-1 optimizer shards on rank 0 (a collective: every rank calls it before a
    save). No-op for other optimizers and without ``save_optimizer_state``."""
    if cfg.save_optimizer_state and isinstance(optimizer, ZeroRedundancyOptimizer):
        torch.cuda.empty_cache()
        optimizer.consolidate_state_dict(to=0)


def _save_checkpoint(
    accelerator: Accelerator,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    cfg: TrainConfig,
    emas: dict[str, EMAModel] | None = None,
) -> None:
    """Save model weights, optimizer, scheduler, and training state.

    Accelerate does not manage the ZeRO-1 optimizer: its consolidated state goes to
    ``ZERO1_OPTIMIZER``. Each entry in ``emas`` is saved as ``<stem>.safetensors`` next to the
    checkpoint.
    """
    save_dir = os.path.join(cfg.output_dir, f"checkpoint-{step}")
    torch.cuda.empty_cache()
    if cfg.save_optimizer_state:
        accelerator.save_state(save_dir)
        if isinstance(optimizer, ZeroRedundancyOptimizer):
            zero1_path = os.path.join(save_dir, ZERO1_OPTIMIZER)
            torch.save(optimizer.state_dict(), zero1_path + ".tmp")
            os.replace(zero1_path + ".tmp", zero1_path)
    unwrapped = accelerator.unwrap_model(policy)
    unwrapped.save_pretrained(save_dir)
    if emas and accelerator.is_main_process:
        import safetensors.torch as st

        for stem, ema in emas.items():
            st.save_file(ema.state_dict(), os.path.join(save_dir, f"{stem}.safetensors"))
            with open(os.path.join(save_dir, f"{stem}_update_count.json"), "w") as f:
                json.dump(ema.update_count, f)
    cfg.save_yaml(os.path.join(save_dir, "train_config.yaml"))
    with open(os.path.join(save_dir, "training_state.json"), "w") as f:
        json.dump({"step": step, "epoch": epoch}, f)
    print(f"Saved checkpoint to {save_dir}")
    if cfg.save_optimizer_state:
        decay_start = (
            cfg.steps - cfg.decay_steps if cfg.lr_schedule == "wsd" and cfg.decay_steps else None
        )
        _prune_stale_optimizer_state(cfg.output_dir, step, also_keep=decay_start)


def _load_checkpoint(
    accelerator: Accelerator,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    emas: dict[str, EMAModel],
    ckpt_dir: str,
) -> tuple[int, int]:
    """Load a checkpoint and return ``(step, epoch)``.

    ZeRO-1 checkpoints: ``load_state`` restores model, scheduler and RNG, then every rank loads
    the full optimizer state and keeps its shard. An optimizer state of the other format (ZeRO-1
    or plain) raises. Checkpoints without optimizer state
    (save_optimizer_state=False, or pruned by _prune_stale_optimizer_state) warm-start from the
    model weights only.
    """
    zero1_path = os.path.join(ckpt_dir, ZERO1_OPTIMIZER)
    saved_zero1 = os.path.exists(zero1_path)
    if saved_zero1 or any(Path(ckpt_dir).glob("optimizer*.bin")):
        # Accelerate skips an optimizer it does not manage: a mismatch would restart Adam.
        if saved_zero1 != isinstance(optimizer, ZeroRedundancyOptimizer):
            raise ValueError(
                f"{ckpt_dir} holds {'ZeRO-1' if saved_zero1 else 'plain'} optimizer state; "
                f"resume with use_zero1={saved_zero1}"
            )
        accelerator.load_state(ckpt_dir)
        if saved_zero1:
            optimizer.load_state_dict(torch.load(zero1_path, map_location="cpu", mmap=True))
    else:
        load_pretrained_weights(accelerator.unwrap_model(policy), ckpt_dir)
        if accelerator.is_main_process:
            print(f"  No optimizer state in {ckpt_dir}; warm-starting weights only")
    for stem, ema in emas.items():
        ema_path = os.path.join(ckpt_dir, f"{stem}.safetensors")
        if os.path.exists(ema_path):
            ema.load_state_dict(load_checkpoint_tensors(ema_path))
        elif accelerator.is_main_process:
            print(f"  No EMA found at {ema_path}, re-initialised from current weights")
    with open(os.path.join(ckpt_dir, "training_state.json")) as f:
        state = json.load(f)
    return state["step"], state["epoch"]


def _resume(
    cfg: TrainConfig,
    accelerator: Accelerator,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    emas: dict[str, EMAModel],
) -> tuple[int, int]:
    """Resume from the latest checkpoint or return the initial training position."""
    ckpt_dir = _find_latest_checkpoint(cfg.output_dir)
    if not ckpt_dir:
        if accelerator.is_main_process:
            print("No checkpoint found, starting from scratch")
        return 0, 0
    step, epoch = _load_checkpoint(accelerator, policy, optimizer, emas, ckpt_dir)
    if accelerator.is_main_process:
        print(f"Resumed from {ckpt_dir} at step {step}, epoch {epoch}")
    # RatioSampler draws from Python's global random state in the main process.
    random.seed(cfg.seed + step)
    return step, epoch


def train(config: str | None = None, resume: bool = False, **kwargs: Any):
    """Train the VLA policy from a ``--config`` yaml plus CLI overrides."""
    resume = fire_bool("resume", resume)
    cfg = TrainConfig.from_config_and_overrides(config, kwargs)
    if not cfg.output_dir:
        raise ValueError("output_dir is required: set it in the yaml or pass --output_dir=…")
    accelerator = build_accelerator(cfg)
    configure_torch(cfg.seed)
    if accelerator.is_main_process:
        os.makedirs(cfg.output_dir, exist_ok=True)
        cfg.save_yaml(os.path.join(cfg.output_dir, "train_config.yaml"))
        cfg.print()
    ds, dataloader = build_dataloader(cfg)
    if accelerator.is_main_process:
        effective_batch_size = (
            cfg.batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps
        )
        epochs = (cfg.steps * effective_batch_size) / len(ds)
        print(f"Cumulative effective dataset size: {len(ds):,}")
        print(f"Effective batch size: {effective_batch_size:,}")
        print(f"Epochs: ({cfg.steps:,} × {effective_batch_size}) / {len(ds):,} = {epochs:.2f}")
    policy = build_policy(cfg, ds)
    if cfg.use_zero1 and accelerator.num_processes == 1:
        raise ValueError("use_zero1 shards the optimizer over ranks: set it False on 1 process")
    optimizer, lr_scheduler = build_optimizer(cfg, policy, zero1=cfg.use_zero1)
    if cfg.use_zero1:
        policy, dataloader, lr_scheduler = accelerator.prepare(policy, dataloader, lr_scheduler)
    else:
        policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            policy, optimizer, dataloader, lr_scheduler
        )
    emas: dict[str, EMAModel] = {}
    if cfg.ema_decay and cfg.ema_decay > 0:
        if accelerator.is_main_process:
            print(f"  EMA enabled (decay={cfg.ema_decay})")
        emas = {"ema": EMAModel(accelerator.unwrap_model(policy), decay=cfg.ema_decay)}
    step, epoch = _resume(cfg, accelerator, policy, optimizer, emas) if resume else (0, 0)
    log = MetricLogger(cfg, accelerator)
    val_batches = prefetch_val_batches(cfg, accelerator) if cfg.val_dataset else None
    if accelerator.is_main_process:
        log.log({"step": step, "git_commit": _git_commit()})
    vwe = (
        None
        if cfg.external_policy is not None
        else accelerator.unwrap_model(policy).model.vlm_with_expert
    )
    if accelerator.is_main_process:
        print("Starting training")

    step_s = data_s = 0.0
    windows = 0
    t_fetch = time.perf_counter()
    while step < cfg.steps:
        if hasattr(dataloader, "set_epoch"):
            dataloader.set_epoch(epoch)
        for batch in dataloader:
            if step >= cfg.steps:
                break
            t_start = time.perf_counter()
            data_s += t_start - t_fetch
            if cfg.external_policy is not None:
                windows += batch[ACTION].shape[0]
            else:
                windows += torch.bincount(batch["group_index"]).shape[0]
            if vwe is not None:
                vwe.collect_gate_stats = (step + 1) % cfg.logging_steps == 0
            policy.train()  # Validation leaves the policy in eval mode.
            loss, loss_dict, grad_norm = _micro_step(
                accelerator, policy, optimizer, batch, cfg.grad_clip
            )
            log.add_micro_batch(loss, loss_dict)
            if accelerator.sync_gradients:
                _finish_optimizer_step(accelerator, policy, lr_scheduler, emas, step)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_fetch = time.perf_counter()
            step_s += t_fetch - t_start
            if not accelerator.sync_gradients:
                continue
            step += 1
            log.add_step(step_s, data_s, windows / cfg.gradient_accumulation_steps)
            step_s = data_s = 0.0
            windows = 0
            if step % cfg.logging_steps == 0:
                gate_stats = vwe.gate_stats if vwe is not None else {}
                log.log_train(step, optimizer.param_groups[0]["lr"], grad_norm, batch, gate_stats)
            if val_batches is not None and step % cfg.val_freq == 0:
                accelerator.wait_for_everyone()
                log.log({"step": step, **_validate(cfg, accelerator, policy, val_batches, emas)})
            if step % cfg.save_steps == 0:
                _consolidate_zero1(cfg, optimizer)
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    _save_checkpoint(accelerator, policy, optimizer, step, epoch, cfg, emas)
            t_fetch = time.perf_counter()
        epoch += 1
    if step % cfg.save_steps != 0:
        _consolidate_zero1(cfg, optimizer)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            _save_checkpoint(accelerator, policy, optimizer, step, epoch, cfg, emas)
    if accelerator.is_main_process:
        print("Training complete.")
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    if int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) != 0:
        logging.getLogger().setLevel(logging.ERROR)
    fire.Fire(train)
