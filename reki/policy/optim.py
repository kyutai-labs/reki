"""Training-side weight updates: the EMA of the weights, the optimizer parameter groups, and
``StochasticRoundingAdamW``."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from reki.policy.modeling import RekiModel
    from reki.vlm.qwen3_5.modeling import Qwen3_5VLMWithExpert


class EMAModel:
    """Exponential moving average of the trainable parameters.

    The shadow is fp32, because bf16 rounding loses the small per-step updates. Under DDP each
    rank has an identical shadow.
    """

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            name: p.detach().to(torch.float32).clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        self.update_count: dict[str, int] = {k: 0 for k in self.shadow}
        self._warned_missing = False
        self._warm_started = False

    @torch.no_grad()
    def update(self, model: torch.nn.Module, step: int | None = None) -> None:
        """Move the fp32 shadows toward ``model``. With ``step``, the decay is capped at
        (1+step)/(10+step) until warm-started."""
        if step is not None and not self._warm_started:
            decay = min(self.decay, (1 + step) / (10 + step))
        else:
            decay = self.decay
        live = dict(model.named_parameters())
        for name, shadow_t in self.shadow.items():
            p = live.get(name)
            if p is None:
                continue
            shadow_t.mul_(decay).add_(p.data.to(torch.float32), alpha=1 - decay)
            self.update_count[name] += 1
        if not self._warned_missing:
            missing = [k for k in self.shadow if k not in live]
            if missing:
                from reki.utils.dist_utils import print_main

                print_main(
                    f"[EMA] WARNING: {len(missing)} shadow keys not in "
                    f"model.named_parameters() — never updated. "
                    f"First 10: {missing[:10]}"
                )
                self._warned_missing = True

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        for name, t in self.shadow.items():
            if name in state:
                if state[name].shape != t.shape:
                    raise ValueError(
                        f"EMA shadow shape mismatch for {name}: checkpoint "
                        f"{tuple(state[name].shape)} vs model {tuple(t.shape)}"
                    )
                t.copy_(state[name].to(t.device, dtype=t.dtype))
        self.update_count = {k: 0 for k in self.shadow}
        self._warned_missing = False
        self._warm_started = True

    @contextlib.contextmanager
    def applied(self, model: torch.nn.Module):
        """Temporarily swap the shadow into ``model``.
        Live weights are backed up to CPU and restored on exit."""
        params = dict(model.named_parameters())
        backup: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, shadow_t in self.shadow.items():
                p = params.get(name)
                if p is None:
                    continue
                backup[name] = p.data.to("cpu", copy=True)
                p.data.copy_(shadow_t.to(p.dtype))
        try:
            yield
        finally:
            with torch.no_grad():
                for name, t in backup.items():
                    params[name].data.copy_(t)


def _decayable_param_ids(policy: torch.nn.Module) -> set[int]:
    """``id(p)`` of the params that get weight decay: the weights of Linear, Conv and
    MultiheadAttention projections. Biases, norm gains, embedding tables and standalone
    parameter tables (e.g. the expert register bank) get no decay. A tied LM
    token embedding gets decay through its lm_head Linear."""
    decay: set[int] = set()
    for mod in policy.modules():
        if isinstance(mod, (torch.nn.Linear, torch.nn.modules.conv._ConvNd)):
            decay.add(id(mod.weight))
        elif isinstance(mod, torch.nn.MultiheadAttention):
            for name in ("in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight"):
                p = getattr(mod, name)
                if p is not None:
                    decay.add(id(p))
    return decay


def build_optimizer_groups(
    policy: RekiModel,
    vlm_with_expert: Qwen3_5VLMWithExpert,
    lr: float,
    weight_decay: float,
    backbone_lr_scale: float = 1.0,
) -> tuple[list[dict], dict[str, int]]:
    """Build the optimizer param groups, each split into a decay and a no-decay half. Group
    order: base, expert, proprio.

    The action expert (when it has trainable params) has its own group at ``lr``.
    ``backbone_lr_scale`` scales only the base group (the VLM plus the action/time projections).
    When it is not 1, the proprio state MLPs have their own group at the full ``lr``. Resuming
    the optimizer state matches groups by position, so do not change this order.

    Returns:
        (param_groups, info) where info maps group names to param counts.
    """
    expert = vlm_with_expert.action_expert
    expert_ids = (
        {id(p) for p in expert.parameters() if p.requires_grad}
        if isinstance(expert, torch.nn.Module)
        else set()
    )
    use_separate_expert_lr = bool(expert_ids)
    decay_ids = _decayable_param_ids(policy)
    base_lr = lr * backbone_lr_scale

    def _split(params: list, group_lr: float, name: str) -> tuple[list[dict], dict[str, int]]:
        decay = [p for p in params if id(p) in decay_ids]
        no_decay = [p for p in params if id(p) not in decay_ids]
        groups = [
            {"params": decay, "lr": group_lr, "weight_decay": weight_decay},
            {"params": no_decay, "lr": group_lr, "weight_decay": 0.0},
        ]
        counts = {
            name: sum(p.numel() for p in decay),
            f"{name} (no_decay)": sum(p.numel() for p in no_decay),
        }
        return groups, counts

    proprio_mods = [vlm_with_expert.state_mlps, policy.model.state_adaln_proj]
    proprio_ids = (
        {id(p) for m in proprio_mods if m is not None for p in m.parameters() if p.requires_grad}
        if backbone_lr_scale != 1.0
        else set()
    )
    use_separate_proprio_lr = bool(proprio_ids)

    expert_c, proprio_c, base_c = [], [], []
    for p in policy.parameters():
        if not p.requires_grad:
            continue
        i = id(p)
        if i in expert_ids:
            expert_c.append(p)
        elif i in proprio_ids:
            proprio_c.append(p)
        else:
            base_c.append(p)

    if not (use_separate_expert_lr or use_separate_proprio_lr):
        return _split(base_c, base_lr, "all")

    param_groups, info = _split(base_c, base_lr, f"base_lr ({backbone_lr_scale}x)")
    if use_separate_expert_lr:
        eg, ei = _split(expert_c, lr, "expert_lr (1.0x)")
        param_groups += eg
        info.update(ei)
    if use_separate_proprio_lr:
        pg, pi = _split(proprio_c, lr, "proprio_lr (1.0x)")
        param_groups += pg
        info.update(pi)
    return param_groups, info


_SR_SEED_BASE = 0x5EED


def _stochastic_round_to_bf16_(
    dst: torch.Tensor, src_fp32: torch.Tensor, generator: torch.Generator | None = None
) -> None:
    """Copy fp32 ``src_fp32`` into bf16 ``dst`` with stochastic rounding.

    bf16 is the top 16 bits of the fp32 word. The function adds uniform random noise to the low
    16 bits, then drops them. The result is unbiased in expectation.

    ``generator`` gives the noise. On plain DDP every rank steps all params, so give every rank
    a generator with the same seed at each step. Otherwise the ranks diverge.

    A NaN whose low 16 payload bits are near 2^16 can wrap to ±0.0. Torch arithmetic does not
    make such NaNs.
    """
    assert src_fp32.dtype == torch.float32
    i = src_fp32.contiguous().view(torch.int32)
    noise = torch.randint(1 << 16, i.shape, generator=generator, device=i.device, dtype=torch.int32)
    rounded = (i + noise) & -0x10000
    dst.copy_(rounded.view(torch.float32))


class StochasticRoundingAdamW(torch.optim.Optimizer):
    """AdamW with the update math in fp32. For an fp32 param it is plain AdamW. For a bf16
    param it writes the result back with stochastic rounding.

    The weights are bf16 by default (``RekiConfig.dtype``). In-place bf16
    AdamW loses small updates: the decay factor ``1 - lr*wd`` rounds to 1.0, and updates below
    2^-8 of the weight vanish. Stochastic rounding keeps them in expectation, with no stored
    fp32 copy. ``master_weights=True`` keeps an fp32 master copy instead.

    ``eps`` defaults to 1e-10, not 1e-8. AdaLN-Zero gates start at zero, so early on sqrt(v̂)
    is about 1e-7 for many coordinates, and an eps of 1e-8 makes the step too small. eps is
    always added in fp32.
    """

    def __init__(
        self,
        params: Any,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-10,
        weight_decay: float = 0.0,
        master_weights: bool = False,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self._sr_step = 0
        self._sr_gens: dict[torch.device, torch.Generator] = {}
        self._master = master_weights
        self._master_validated = False

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the state and keep the moments in fp32.

        The base ``load_state_dict`` casts the moments to the param dtype (bf16 for bf16
        params). This method copies the saved moments in again as fp32.

        When master mode is off, it drops ``master``. Steps without master mode make the
        master stale, and a later master-mode resume would roll the weights back."""
        super().load_state_dict(state_dict)
        saved_ids = [i for g in state_dict["param_groups"] for i in g["params"]]
        current = [p for g in self.param_groups for p in g["params"]]
        id_map = dict(zip(saved_ids, current))
        keys = ("exp_avg", "exp_avg_sq", "master") if self._master else ("exp_avg", "exp_avg_sq")
        for idx, saved in state_dict["state"].items():
            p = id_map[idx]
            self.state[p].pop("master", None)
            for key in keys:
                if key in saved:
                    self.state[p][key] = saved[key].to(
                        device=p.device, dtype=torch.float32, copy=True
                    )

    @torch.no_grad()
    def step(  # type: ignore[override]
        self, closure: Callable[[], float] | None = None
    ) -> float | None:
        """One AdamW step in fp32; bf16 params are written back as in the class docstring."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._sr_step += 1
        seeded: set[torch.device] = set()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("StochasticRoundingAdamW does not support sparse gradients")
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                grad = p.grad.float()
                if self._master and p.dtype == torch.bfloat16:
                    if "master" not in state:
                        state["master"] = p.detach().to(torch.float32, copy=True)
                    elif not self._master_validated and not torch.equal(
                        state["master"].to(p.dtype), p
                    ):
                        raise RuntimeError(
                            "StochasticRoundingAdamW: restored fp32 master does not round to "
                            "the current bf16 param — p moved under a different write-back, so "
                            "stepping from this master would roll those updates back."
                        )
                    p_fp32 = state["master"]
                else:
                    p_fp32 = p.float()
                if weight_decay != 0:
                    p_fp32.mul_(1 - lr * weight_decay)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                denom = (exp_avg_sq / bias_correction2).sqrt_().add_(eps)
                p_fp32.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)

                if p.dtype == torch.float32:
                    p.copy_(p_fp32)
                elif p.dtype == torch.bfloat16 and self._master:
                    p.copy_(p_fp32)
                elif p.dtype == torch.bfloat16:
                    gen = self._sr_gens.get(p.device)
                    if gen is None:
                        gen = torch.Generator(device=p.device)
                        self._sr_gens[p.device] = gen
                    if p.device not in seeded:
                        gen.manual_seed(_SR_SEED_BASE + self._sr_step)
                        seeded.add(p.device)
                    _stochastic_round_to_bf16_(p, p_fp32, generator=gen)
                else:
                    raise RuntimeError(
                        f"StochasticRoundingAdamW: unsupported param dtype {p.dtype}"
                    )
        self._master_validated = True
        return loss
