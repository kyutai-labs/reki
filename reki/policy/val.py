"""Validation metric: the flow loss on fixed val batches."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, SubsetRandomSampler

from reki.data.robotics.collate import combined_collate
from reki.data.robotics.factory import make_val_dataset
from reki.policy.external import get_external_policy
from reki.policy.utils import to_device

if TYPE_CHECKING:
    from reki.policy.train_config import TrainConfig


def prefetch_val_batches(cfg: "TrainConfig", accelerator: Accelerator | None = None) -> list[dict]:
    """Build a fixed list of val batches once (shared across all val events). The collate does
    not trim: each batch holds ``val_batch_size`` samples."""
    assert cfg.val_dataset
    is_main = accelerator is None or accelerator.is_main_process
    if is_main:
        print(f"Building val dataset: {cfg.val_dataset}")
    val_ds = make_val_dataset(cfg, cfg.val_dataset)
    rank = accelerator.process_index if accelerator is not None else 0
    gen = torch.Generator().manual_seed(cfg.val_seed + rank)
    held_out = val_ds.val_frame_indices
    sampler = SubsetRandomSampler(held_out, generator=gen) if held_out else None
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.val_batch_size,
        shuffle=None if sampler is not None else True,
        sampler=sampler,
        generator=gen,
        num_workers=min(cfg.num_workers, 2),
        collate_fn=(
            get_external_policy(cfg.external_policy).collate
            if cfg.external_policy is not None
            else combined_collate
        ),
        drop_last=True,
        pin_memory=True,
    )
    it = iter(val_loader)
    batches = list(itertools.islice(it, cfg.val_batches))
    if len(batches) < cfg.val_batches:
        raise ValueError(
            f"val set gives {len(batches)} batches of {cfg.val_batch_size} per rank, "
            f"val_batches={cfg.val_batches}"
        )
    if is_main:
        print(f"Cached {len(batches)} val batches per rank (batch_size={cfg.val_batch_size})")
    return batches


def val_flow_loss(
    policy: torch.nn.Module,
    val_batches: list[dict],
    device: torch.device,
    seed: int,
    accelerator: Accelerator | None = None,
) -> torch.Tensor:
    """Training flow-matching loss on the val batches (pointwise v_t MSE).

    One forward per batch; sampling of t and noise is seeded for reproducibility.
    """
    policy.eval()
    losses: list[torch.Tensor] = []
    # Fork so the seeded val draws do not reset the training RNG.
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
        torch.manual_seed(seed)  # also seeds every CUDA device
        for batch in val_batches:
            batch_dev = to_device(batch, device)
            autocast_ctx = (
                accelerator.autocast()
                if accelerator
                else torch.autocast("cuda", dtype=torch.bfloat16)
            )
            with autocast_ctx:
                _, loss_dict = policy(batch_dev)
            losses.append(loss_dict["flow_loss"].detach().float())
    local = torch.stack(losses).mean()
    if accelerator is not None:
        local = accelerator.gather(local.unsqueeze(0)).mean()
    return local
