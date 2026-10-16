"""Gradient accumulation in ``scripts/train.py``: with accum=k, one optimizer step per k
micro-batches (clip, optimizer, LR schedule, EMA), and the update equals one step on the
mean of the k micro-batch gradients."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState, GradientState
from accelerate.utils import GradientAccumulationPlugin

pytestmark = pytest.mark.cpu

_spec = importlib.util.spec_from_file_location(
    "reki_train_script", Path(__file__).parents[2] / "scripts" / "train.py"
)
assert _spec is not None and _spec.loader is not None
train_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_script)

ACCUM, N_MICRO, CLIP = 2, 6, 0.05


class _Toy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(3, 2)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
        loss = ((self.lin(batch["x"]) - batch["y"]) ** 2).mean()
        return loss, {}


class _CountingEMA:
    def __init__(self) -> None:
        self.steps: list[int] = []

    def update(self, model: torch.nn.Module, step: int) -> None:
        self.steps.append(step)


@pytest.fixture
def accelerator():
    AcceleratorState._reset_state(True)
    GradientState._reset_state()
    acc = Accelerator(
        cpu=True,
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=ACCUM, sync_with_dataloader=False
        ),
        step_scheduler_with_optimizer=False,
    )
    yield acc
    AcceleratorState._reset_state(True)
    GradientState._reset_state()


def test_accumulation_steps_once_per_group_on_mean_gradient(accelerator: Accelerator):
    torch.manual_seed(0)
    batches = [{"x": torch.randn(4, 3), "y": torch.randn(4, 2)} for _ in range(N_MICRO)]
    model = _Toy()
    ref = _Toy()
    ref.load_state_dict(model.state_dict())

    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0 / (1 + s))
    model, opt, sched = accelerator.prepare(model, opt, sched)
    ema = _CountingEMA()

    clip_calls = 0
    real_clip = accelerator.clip_grad_norm_

    def counting_clip(params: Any, max_norm: float) -> Any:
        nonlocal clip_calls
        clip_calls += 1
        return real_clip(params, max_norm)

    accelerator.clip_grad_norm_ = counting_clip

    step = 0
    grad_norms = []
    for batch in batches:
        _, _, grad_norm = train_script._micro_step(accelerator, model, opt, batch, CLIP)
        if not accelerator.sync_gradients:
            assert grad_norm is None
            continue
        grad_norms.append(grad_norm)
        train_script._finish_optimizer_step(accelerator, model, sched, {"ema": ema}, step)
        step += 1

    n_opt = N_MICRO // ACCUM
    assert step == n_opt
    assert clip_calls == n_opt
    assert ema.steps == list(range(n_opt))
    assert sched.scheduler.last_epoch == n_opt
    assert all(g is not None for g in grad_norms)

    ref_opt = torch.optim.SGD(ref.parameters(), lr=0.1)
    ref_sched = torch.optim.lr_scheduler.LambdaLR(ref_opt, lambda s: 1.0 / (1 + s))
    for i in range(n_opt):
        group = batches[i * ACCUM : (i + 1) * ACCUM]
        loss = torch.stack([ref(b)[0] for b in group]).mean()
        ref_opt.zero_grad()
        loss.backward()
        ref_norm = torch.nn.utils.clip_grad_norm_(ref.parameters(), CLIP)
        assert ref_norm > CLIP
        assert torch.allclose(grad_norms[i], ref_norm, atol=1e-6)
        ref_opt.step()
        ref_sched.step()

    for p, q in zip(model.parameters(), ref.parameters()):
        assert torch.allclose(p, q, atol=1e-6)
