"""Inspect data and exercise the VLA training pipeline with detailed diagnostics.

Builds the configured dataset, policy, optimizer, and scheduler, then runs forward and backward
passes while reporting batch contents, timings, gradients, and GPU memory.

Usage:
    pixi run python scripts/dry_run.py --config=configs/sft.yaml --dataset=libero --batch_size=4
    pixi run python scripts/dry_run.py --config=configs/sft.yaml --dataset=libero --skip_model

Dry-run-only knobs: ``device``, ``dtype``, ``num_batches``, ``skip_model``, ``skip_backward``,
``debug_positions``. Everything else is a TrainConfig field.
"""

from __future__ import annotations

import textwrap
import time
from typing import Any

import fire
import torch
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_TOKENS,
)
from train import build_dataloader, build_optimizer, build_policy, configure_torch
from transformers import AutoProcessor

from reki.policy.modeling import RekiModel
from reki.policy.train_config import TrainConfig
from reki.policy.utils import to_device
from reki.utils.cli import fire_bool


def format_sample_tokens(ids: list[int], tokenizer: object, image_pad_id: int | None) -> str:
    """Decode a token sequence in order, collapsing IMAGE_PAD runs."""
    parts: list[str] = []
    n = len(ids)
    i = 0
    while i < n:
        j = i
        if image_pad_id is not None and ids[i] == image_pad_id:
            while j < n and ids[j] == image_pad_id:
                j += 1
            parts.append(f"\033[1;34m[IMAGE_PAD×{j - i}]\033[0m")
        else:
            while j < n and not (image_pad_id is not None and ids[j] == image_pad_id):
                j += 1
            parts.append(tokenizer.decode(ids[i:j]))  # type: ignore[union-attr]
        i = j
    return "".join(parts)


def _shape_str(v: object) -> str:
    """Human-readable shape / type string for a batch value."""
    if isinstance(v, torch.Tensor):
        return f"Tensor {tuple(v.shape)}  dtype={v.dtype}  device={v.device}"
    if isinstance(v, list):
        descs = []
        for i, el in enumerate(v):
            if el is None:
                descs.append("  None")
            elif isinstance(el, torch.Tensor):
                descs.append(f"  [{i}] Tensor {tuple(el.shape)}  dtype={el.dtype}")
            else:
                descs.append(f"  [{i}] {type(el).__name__}")
            if i >= 5:
                descs.append(f"  ... ({len(v)} items total)")
                break
        return "list[\n" + "\n".join(descs) + "\n]"
    return repr(v)[:120]


def _section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _print_batch_diagnostics(batch: dict, cfg: TrainConfig) -> None:
    """Token counts, batch contents and decoded prompts of one collated batch."""
    if cfg.external_policy is not None:
        _section("BATCH CONTENTS")
        for key in sorted(batch.keys()):
            print(f"\n  {key}:")
            print(textwrap.indent(_shape_str(batch[key]), "    "))
        return

    _section("TOKEN COUNTS")
    packed_tokens = batch[OBS_LANGUAGE_TOKENS]
    seq_lens = batch["seq_lens"]
    B = seq_lens.shape[0]
    total_text = int(seq_lens.sum().item())
    from reki.data.robotics.tokenization import count_vision_tokens

    img_tokens = sum(count_vision_tokens(vi) for vi in batch.get("vision_inputs", []))
    actions = batch[ACTION]
    n_robot = sum(1 for a in actions if a is not None)
    pads = batch.get("action_is_pad")
    if pads is not None and isinstance(pads, list):
        full_pad_count = sum(1 for p in pads if p is not None and bool(p.all().item()))
        any_pad_count = sum(1 for p in pads if p is not None and bool(p.any().item()))
        avg_pad_frac = sum(float(p.float().mean().item()) for p in pads if p is not None) / max(
            1, n_robot
        )
        print(
            f"  action_is_pad:        full_pad={full_pad_count}/{n_robot}, any_pad={any_pad_count}/{n_robot}, avg_pad_frac={avg_pad_frac:.2%}"
        )
        if full_pad_count > 0:
            print(
                "    WARNING: some samples have only padded action targets: check the "
                "dataset's index column and action windows."
            )
    print(f"  Batch size:           {B}")
    print(f"  Robotics samples:     {n_robot}")
    print(f"  Text tokens (total):  {total_text}")
    print(f"  Per-sample seq_lens:  {seq_lens.tolist()}")
    if "k_eff" in batch and batch["k_eff"]:
        k_effs = batch["k_eff"]
        print(
            f"  Per-sample k_eff:     {k_effs}  (min={min(k_effs)} max={max(k_effs)} avg={sum(k_effs) / len(k_effs):.2f})"
        )
    print(f"  Image tokens (total): {img_tokens}")
    print(f"  State encoding:       {cfg.state_encoding}")

    _section("BATCH CONTENTS")
    for key in sorted(batch.keys()):
        print(f"\n  {key}:")
        print(textwrap.indent(_shape_str(batch[key]), "    "))

    processor = AutoProcessor.from_pretrained(cfg.vlm_model_id)
    tokenizer = processor.tokenizer
    _img_pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if _img_pad in (None, tokenizer.unk_token_id):
        _img_pad = None
    _section("DECODED LANGUAGE TOKENS")
    cu = torch.zeros(B + 1, dtype=torch.int32)
    cu[1:] = seq_lens.cumsum(0)
    for i in range(B):
        start, end = int(cu[i]), int(cu[i + 1])
        ids = packed_tokens[start:end].tolist()
        print(f"\n  Sample {i} ({len(ids)} tok):")
        print(f"    {format_sample_tokens(ids, tokenizer, _img_pad)}")


def _shape(t: object) -> str:
    if isinstance(t, torch.Tensor):
        base = f"Tensor{tuple(t.shape)}"
        if t.numel() == 0 or t.dim() == 0:
            return base
        if t.numel() <= 16 and t.dtype in (torch.int32, torch.int64, torch.bool):
            return f"{base}={t.flatten().tolist()}"
        if t.dim() == 1 and t.numel() <= 64:
            return f"{base} range=[{t.min().item():.4g}, {t.max().item():.4g}]"
        return base
    if isinstance(t, tuple):
        return "(" + ", ".join(_shape(x) for x in t) + ")"
    if isinstance(t, list):
        return "[" + ", ".join(_shape(x) for x in t[:3]) + ("…" if len(t) > 3 else "") + "]"
    if t is None:
        return "None"
    return type(t).__name__


def _instrument_layer0(vwe: Any) -> list[Any]:
    """Print block I/O (args+kwargs shapes) and every nn.Linear i/o inside the first VLM and
    expert layer. Returns the hook handles."""

    def _make_proj_hook(label: str):
        def hook(_module: torch.nn.Module, args: tuple[torch.Tensor, ...], output: object) -> None:
            inp_shape = _shape(args[0]) if args else "?"
            print(f"    {label}: input {inp_shape} -> output {_shape(output)}")

        return hook

    def _make_block_hook(label: str):
        def hook(
            _module: torch.nn.Module,
            args: tuple[object, ...],
            kwargs: dict[str, object],
            output: object,
        ) -> None:
            print(f"\n  [{label}] input:")
            for i, a in enumerate(args):
                print(f"    args[{i}]: {_shape(a)}")
            for k, v in kwargs.items():
                print(f"    {k}: {_shape(v)}")
            print(f"  [{label}] output: {_shape(output)}")

        return hook

    def _first_layer(module: object) -> torch.nn.Module | None:
        if not isinstance(module, torch.nn.Module):
            return None
        layers = getattr(module, "layers", None)
        if isinstance(layers, torch.nn.ModuleList) and len(layers) > 0:
            return layers[0]
        inner = getattr(module, "model", None) or getattr(module, "language_model", None)
        return _first_layer(inner)

    hooks: list[Any] = []
    for tag, root in (("vlm", vwe.vlm), ("expert", vwe.action_expert)):
        if root is None:
            continue
        layer0 = _first_layer(root)
        if layer0 is None:
            continue
        hooks.append(
            layer0.register_forward_hook(_make_block_hook(f"{tag}.layer0"), with_kwargs=True)
        )
        for name, sub in layer0.named_modules():
            if isinstance(sub, torch.nn.Linear):
                hooks.append(sub.register_forward_hook(_make_proj_hook(f"{tag}.{name}")))
    return hooks


def _print_positions(vwe: Any, batch: dict, chunk_size: int) -> None:
    _section("MRoPE POSITIONS")
    pos_ids = vwe.fwd.position_ids
    starts = vwe.fwd.expert_starts
    if pos_ids is None or starts is None:
        print("  (no position_ids / expert_starts in the forward state)")
        return
    seq_lens_list = batch["seq_lens"].tolist()
    cu_local = [0]
    for s in seq_lens_list:
        cu_local.append(cu_local[-1] + s)
    print("  per-sample (seq_len = token count; expert_start = Qwen's natural next position):")
    for i, (s, start) in enumerate(zip(seq_lens_list, starts)):
        s0, s1 = cu_local[i], cu_local[i + 1]
        sample_pos = pos_ids[..., s0:s1]
        t_max = int(sample_pos[0].max().item())
        h_max = int(sample_pos[1].max().item())
        w_max = int(sample_pos[2].max().item())
        print(
            f"    sample {i}: seq_len={s}  expert_start={start}  "
            f"gap=seq_len-start={s - start}  "
            f"max_prefix_pos(T,H,W)=({t_max},{h_max},{w_max})"
        )
        vis_t = sample_pos[0, 0, :].tolist()
        samples_at = [0, min(5, s - 1), s // 2, max(0, s - 2), s - 1]
        snaps = [(j, vis_t[j]) for j in samples_at]
        print(f"      T at idx {snaps}")
        print(f"      expert action_0..{chunk_size - 1} → T=H=W={start}..{start + chunk_size - 1}")


def _print_gradient_diagnostics(policy: torch.nn.Module, cfg: TrainConfig) -> None:
    grad_groups: dict[str, list[float]] = {}
    for name, param in policy.named_parameters():
        if param.grad is not None:
            parts = name.split(".")
            group = ".".join(parts[: min(3, len(parts))])
            grad_groups.setdefault(group, []).append(param.grad.norm().item())

    print("\n  Gradient norms (L2, grouped):")
    for group in sorted(grad_groups):
        norms = grad_groups[group]
        total_norm = (sum(n**2 for n in norms)) ** 0.5
        print(f"    {group}: {total_norm:.6f}  ({len(norms)} params)")

    nan_params = [
        name
        for name, p in policy.named_parameters()
        if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
    ]
    if nan_params:
        print(f"\n  WARNING: NaN/Inf gradients in: {nan_params}")
    else:
        print("\n  No NaN/Inf gradients.")

    ddp_unused = [n for n, p in policy.named_parameters() if p.requires_grad and p.grad is None]
    if ddp_unused:
        print(f"\n  {len(ddp_unused)} trainable params with no grad:")
        for n in ddp_unused[:10]:
            print(f"    {n}")
        if len(ddp_unused) > 10:
            print(f"    ... and {len(ddp_unused) - 10} more")
        if cfg.ddp_find_unused_parameters:
            print("  DDP uses find_unused_parameters=True, so these are allowed.")
        else:
            raise RuntimeError(
                f"DDP would crash: {len(ddp_unused)} trainable params have no gradient. "
                "Either freeze them, gate the module behind a different flag, or wire "
                "them into the forward path."
            )
    else:
        n_trainable = sum(1 for p in policy.parameters() if p.requires_grad)
        print(f"\n  DDP-readiness: all {n_trainable} trainable params received gradients ✓")


def _optimizer_step(
    policy: torch.nn.Module, optimizer: Any, lr_scheduler: Any, grad_clip: float
) -> float | None:
    grad_norm = None
    if grad_clip > 0:
        grad_norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip))
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return grad_norm


def dry_run(
    config: str | None = None,
    device: str = "cuda",
    dtype: str = "bfloat16",
    num_batches: int = 1,
    skip_backward: bool = False,
    skip_model: bool = False,
    debug_positions: bool = False,
    **overrides: Any,
):
    """Inspect batches and run the configured policy for ``num_batches`` iterations.

    Args:
        config: ``--config`` yaml, same loader as scripts/train.py; ``overrides`` are
            TrainConfig fields merged on top (e.g. ``--dataset=libero --batch_size=4``).
        device: where the model and batches live.
        dtype: model weight dtype; the forward autocasts to it.
        num_batches: batches to fetch; diagnostics print for the first, later ones train.
        skip_backward: forward only.
        skip_model: skip model loading entirely (only inspect data).
        debug_positions: print the MRoPE position layout after the first forward.
    """
    skip_backward = fire_bool("skip_backward", skip_backward)
    skip_model = fire_bool("skip_model", skip_model)
    debug_positions = fire_bool("debug_positions", debug_positions)
    cfg = TrainConfig.from_config_and_overrides(config, overrides)
    configure_torch(cfg.seed)
    cfg.print()

    _section("LOADING DATASET")
    ds, _, loader = build_dataloader(cfg)
    print(f"Dataset length: {len(ds):,}")

    _section("FETCHING BATCH")
    batch_iter = iter(loader)
    batch: Any = next(batch_iter)
    _print_batch_diagnostics(batch, cfg)

    if skip_model:
        _section("DRY RUN COMPLETE (model skipped)")
        return

    _section("BUILDING POLICY")
    torch_device = torch.device(device)
    policy = build_policy(cfg, ds, device=device, dtype=dtype).to(torch_device)

    total = sum(p.numel() for p in policy.parameters())
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"  Total params:     {total:>12,}")
    print(f"  Trainable params: {trainable:>12,}")
    print(f"  Frozen params:    {total - trainable:>12,}")

    vwe: Any = None
    if isinstance(policy, RekiModel):
        vwe = policy.model.vlm_with_expert
        vlm_params = sum(p.numel() for p in vwe.vlm.parameters())
        print(f"  VLM params:       {vlm_params:>12,}")
        ae_params = sum(p.numel() for p in vwe.action_expert.parameters())
        print(
            f"  Action expert:    {ae_params:>12,}  "
            f"(hidden={policy.config.action_expert_hidden_size}, "
            f"intermediate={policy.config.action_expert_intermediate_size})"
        )

    optimizer, lr_scheduler = build_optimizer(cfg, policy)
    torch_dtype = getattr(torch, dtype)
    batch = to_device(batch, torch_device)
    _hooks = []
    if isinstance(policy, RekiModel):
        _hooks = _instrument_layer0(vwe)
        _section("INSTRUMENTING LAYER 0")

    def _sync() -> None:
        if torch_device.type == "cuda":
            torch.cuda.synchronize()

    _section("FORWARD PASS")
    policy.train()
    if torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    _sync()
    t0 = time.perf_counter()
    with torch.autocast(torch_device.type, dtype=torch_dtype):
        loss, loss_dict = policy.forward(batch)
    _sync()
    fwd_ms = 1000 * (time.perf_counter() - t0)
    for h in _hooks:
        h.remove()

    if debug_positions and isinstance(policy, RekiModel):
        _print_positions(vwe, batch, policy.config.chunk_size)

    print(f"\n  Loss: {loss.item():.6f}")
    for k, v in loss_dict.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.item():.6f}" if v.dim() == 0 else f"  {k}: {tuple(v.shape)}")
        else:
            print(f"  {k}: {v}")

    if skip_backward:
        print(f"\n  Forward: {fwd_ms:.1f} ms")
        _section("DRY RUN COMPLETE (backward skipped)")
        return

    _section("BACKWARD PASS")
    _sync()
    t0 = time.perf_counter()
    loss.backward()
    _sync()
    bwd_ms = 1000 * (time.perf_counter() - t0)
    print(
        f"\n  Forward: {fwd_ms:.1f} ms, Backward: {bwd_ms:.1f} ms, Total: {fwd_ms + bwd_ms:.1f} ms"
    )
    _print_gradient_diagnostics(policy, cfg)
    if num_batches > 1:
        _section(f"FURTHER BATCHES ({num_batches - 1} × fwd+bwd+step)")
        grad_norm = _optimizer_step(policy, optimizer, lr_scheduler, cfg.grad_clip)
        print(f"  batch  0: grad_norm={'-' if grad_norm is None else f'{grad_norm:.3f}'}")
        for i in range(1, num_batches):
            batch = to_device(next(batch_iter), torch_device)
            _sync()
            t0 = time.perf_counter()
            with torch.autocast(torch_device.type, dtype=torch_dtype):
                loss_i, _ = policy.forward(batch)
            _sync()
            t_f = 1000 * (time.perf_counter() - t0)
            t0 = time.perf_counter()
            loss_i.backward()
            _sync()
            t_b = 1000 * (time.perf_counter() - t0)
            t0 = time.perf_counter()
            grad_norm = _optimizer_step(policy, optimizer, lr_scheduler, cfg.grad_clip)
            _sync()
            t_s = 1000 * (time.perf_counter() - t0)
            line = (
                f"  batch {i:2d}: loss={float(loss_i):.4f}  "
                f"grad_norm={'-' if grad_norm is None else f'{grad_norm:.3f}'}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}  fwd={t_f:6.1f}  bwd={t_b:7.1f}  "
                f"step={t_s:7.1f} ms"
            )
            if "seq_lens" in batch:
                line += f"  n_samples={int(batch['seq_lens'].shape[0])}"
            print(line)

    if torch_device.type == "cuda":
        _section("GPU MEMORY")
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            peak = torch.cuda.max_memory_allocated(i) / 1024**3
            print(
                f"  GPU {i}: allocated={alloc:.2f} GiB, "
                f"reserved={reserved:.2f} GiB, peak_alloc(fwd+bwd)={peak:.2f} GiB"
            )

    _section("DRY RUN COMPLETE")


if __name__ == "__main__":
    fire.Fire(dry_run)
