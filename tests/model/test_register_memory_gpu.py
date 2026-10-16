"""Register memory end to end on a tiny random Qwen3.5 (``tiny_qwen``) in the shipped regime:
adaln proprio, recurrent registers, K=3, a synthetic libero group (``synthetic_group``).

- streaming inference (one frame per predict through the RegisterCache) reproduces the batched
  training forward, and feeds every Euler step zero padded action dims;
- the prefix/register cache built once per frame stays valid across all Euler steps;
- the vectorized multi-draw loss equals the mean of single-draw losses;
- frame causality: a frame's noise / τ / padded-dim noise moves no other frame's loss;
- every frame is supervised and the register bank and the proprio projection get gradients;
- a state-dropped sample's AdaLN cond is the learned null (``state_adaln_proj.bias``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

pytestmark = [pytest.mark.gpu, pytest.mark.usefixtures("tiny_qwen")]

SEED = 0
DEVICE = torch.device("cuda")


def _open_gates(policy: Any) -> None:
    """AdaRMS-Zero cold init zeroes every gate (and the cond weights), muting the expert's
    attention/MLP outputs and the τ/state cond at step 0. Emulate a trained expert: open the
    gate bias, give the cond weights and the zero-init proprio projection a small signal."""
    for layer in policy.model.vlm_with_expert.action_expert.layers:
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "cross_attn_layernorm",
        ):
            norm = getattr(layer, name, None)
            if norm is None or getattr(norm, "dense", None) is None:
                continue
            with torch.no_grad():
                norm.dense.weight.normal_(std=0.02)
                norm.dense.bias[2 * norm.dim :] = 0.5
    with torch.no_grad():
        policy.model.state_adaln_proj.weight.normal_(std=0.02)


def _policy_and_batch(
    synthetic_group: Callable[..., tuple[Any, dict]], **overrides: Any
) -> tuple[Any, dict, dict]:
    from reki.data.robotics.collate import combined_collate
    from reki.policy.configuration import RekiConfig
    from reki.policy.modeling import RekiModel
    from reki.policy.utils import to_device

    cfg, group = synthetic_group(k=3, **overrides)
    torch.manual_seed(SEED)
    policy = RekiModel(RekiConfig.from_train_config(cfg, device="cuda")).to(DEVICE).train()
    _open_gates(policy)
    batch = to_device(combined_collate([group], max_text_tokens=cfg.max_text_tokens), DEVICE)
    return policy, batch, group


def _frame_bounds(policy: Any, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    actions, _pad, _ri, _af, cs = policy.prepare_action(batch)
    cu = torch.zeros(cs.shape[0] + 1, dtype=torch.long, device=DEVICE)
    cu[1:] = cs.to(DEVICE).long().cumsum(0)
    return actions, cu


def _per_frame_flow_loss(policy: Any, batch: dict) -> torch.Tensor:
    """[n_frames] flow loss of each frame: the masked per-element loss of the model forward,
    averaged over each frame's real (action row, dim) entries."""
    out: dict[str, Any] = {}
    model_forward = policy.model.forward

    def capture(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        res = model_forward(*args, **kwargs)
        out.update(kwargs, flow_loss=res["flow_loss"])
        return res

    policy.model.forward = capture
    try:
        policy.forward(batch)
    finally:
        del policy.model.forward
    mask = out["action_dim_mask"].to(out["flow_loss"].dtype)
    cs = out["cs_per_sample"]
    frame = torch.arange(cs.shape[0], device=cs.device).repeat_interleave(cs)
    num = torch.zeros(cs.shape[0], device=cs.device).index_add_(
        0, frame, (out["flow_loss"] * mask).sum(-1).float()
    )
    den = torch.zeros(cs.shape[0], device=cs.device).index_add_(0, frame, mask.sum(-1).float())
    return num / den.clamp_min(1)


def _per_frame_losses(policy: Any, batch: dict, noise: torch.Tensor, time: torch.Tensor):
    policy.model.sample_noise = lambda shape, dev, _n=noise: _n
    policy.model.sample_time = lambda b, dev, _t=time: _t
    return _per_frame_flow_loss(policy, batch).detach()


def _member_infer_batch(member: dict) -> dict:
    """The single-frame inference batch ``predict_action_chunk`` receives for one frame."""
    from reki.policy.utils import to_device

    m = to_device(dict(member), DEVICE)
    tokens = m[OBS_LANGUAGE_TOKENS]
    return {
        OBS_LANGUAGE_TOKENS: tokens,
        "vision_inputs": [m["vision_inputs"]],
        "k_eff": [1],
        "action_dim": m[ACTION].shape[-1],
        "state_mlp_embodiment": [m["state_mlp_embodiment"]],
        OBS_STATE: [m[OBS_STATE]],
    }


def test_register_streaming_vs_batch_parity(synthetic_group: Callable[..., Any]) -> None:
    """K sequential predicts (building the register ring) reproduce the last frame's velocity
    from one batched training forward at a matched (x_t, τ=1). The episode fits the window, so
    the recurrent registers see the same frames on both paths; the residual is bf16 rounding."""
    import reki.policy.modeling as M

    policy, batch, group = _policy_and_batch(synthetic_group)
    model = policy.model
    actions, cu = _frame_bounds(policy, batch)
    last = slice(int(cu[-2]), int(cu[-1]))
    torch.manual_seed(SEED)
    noise = torch.randn(actions.shape, device=DEVICE)
    time = torch.full((cu.shape[0] - 1,), 0.5, device=DEVICE)
    time[-1] = 1.0

    cap: dict[str, torch.Tensor] = {}
    proj = M._proj_fp32

    def capture(module: Any, x: torch.Tensor) -> torch.Tensor:
        out = proj(module, x)
        if module is model.action_out_proj:
            cap["v"] = out.detach().float()
        return out

    def stream(members: list[dict]) -> torch.Tensor:
        policy.reset_episode()
        for f, member in enumerate(members):
            if f == len(members) - 1:
                model.sample_noise = lambda shape, dev, _n=noise[last].unsqueeze(0): _n
            else:
                model.sample_noise = lambda shape, dev: torch.randn(shape, device=dev)
            policy.predict_action_chunk(_member_infer_batch(member))
        return cap["v"].clone()

    M._proj_fp32 = capture  # type: ignore[assignment]
    try:
        model.sample_noise = lambda shape, dev: noise
        model.sample_time = lambda b, dev: time
        policy.forward(batch)
        v_batch = cap["v"][last].clone()

        policy.eval()
        policy.config.num_inference_steps = 1
        members = group["group_items"]
        v_stream = stream(members)
        v_no_history = stream(members[-1:])
    finally:
        M._proj_fp32 = proj  # type: ignore[assignment]

    scale = v_batch.abs().max().clamp_min(1.0).item()
    diff = (v_stream - v_batch).abs().max().item()
    assert diff < 3e-2 * scale, f"streaming vs batched last-frame velocity max|Δ|={diff}"
    history_effect = (v_stream - v_no_history).abs().max().item()
    assert history_effect > 10 * diff, f"history moves v by only {history_effect} ({diff=})"

    seen: list[torch.Tensor] = []
    embed = model._action_embs

    def spy(x_t: torch.Tensor) -> torch.Tensor:
        seen.append(x_t.detach().clone())
        return embed(x_t)

    model._action_embs = spy  # type: ignore[method-assign]
    policy.config.num_inference_steps = 4
    policy.reset_episode()
    policy.predict_action_chunk(_member_infer_batch(members[0]))
    action_dim = members[0][ACTION].shape[-1]
    assert len(seen) == 4 and seen[0].shape[-1] > action_dim
    assert all(bool((x[..., action_dim:] == 0).all()) for x in seen)


def test_register_flow_draws_parity(synthetic_group: Callable[..., Any]) -> None:
    """The vectorized k-draw loss (owner-major tiling, draw-shared registers) equals the mean
    of k single-draw losses, k = the shipped flow_draws=8."""
    policy, batch, _ = _policy_and_batch(synthetic_group)
    model = policy.model
    actions, cu = _frame_bounds(policy, batch)
    n_frames, k = cu.shape[0] - 1, 8
    torch.manual_seed(SEED)
    noises = [torch.randn(actions.shape, device=DEVICE) for _ in range(k)]
    times = [torch.rand(n_frames, device=DEVICE) for _ in range(k)]

    def owner_major(ns: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat([n[int(cu[i]) : int(cu[i + 1])] for i in range(n_frames) for n in ns], 0)

    queue_n: list[torch.Tensor] = []
    queue_t: list[torch.Tensor] = []
    model.sample_noise = lambda shape, dev: queue_n.pop(0)
    model.sample_time = lambda b, dev: queue_t.pop(0)

    def flow_loss(ns: list[torch.Tensor], ts: list[torch.Tensor], draws: int) -> float:
        policy.config.flow_draws = draws
        queue_n[:], queue_t[:] = ns, ts
        _, ld = policy.forward(batch)
        assert not queue_n and not queue_t
        return float(ld["flow_loss"])

    vec = flow_loss([owner_major(noises)], [torch.stack(times, 1).reshape(-1)], k)
    cs = model.vlm_with_expert.fwd.cs_per_sample
    n_reg = policy.config.action_expert_num_registers
    assert int(cs.sum()) == n_reg * n_frames + k * actions.shape[0], "registers once per owner"
    loop = sum(flow_loss([n], [t], 1) for n, t in zip(noises, times)) / k
    assert abs(vec - loop) < 1e-2 * max(1.0, loop), f"vectorized {vec} != loop-mean {loop}"


def test_register_frame_causality(synthetic_group: Callable[..., Any]) -> None:
    """Registers never see noisy actions, so frame f's noise or τ moves frame f's loss only,
    and the padded action dims carry no noise and no loss at all."""
    policy, batch, group = _policy_and_batch(synthetic_group)
    actions, cu = _frame_bounds(policy, batch)
    torch.manual_seed(SEED)
    base = torch.randn(actions.shape, device=DEVICE)
    t = torch.full((cu.shape[0] - 1,), 0.5, device=DEVICE)
    ref = _per_frame_losses(policy, batch, base, t)

    def moved(alt: torch.Tensor) -> list[bool]:
        return [abs(float(a - r)) > 1e-5 for a, r in zip(alt, ref)]

    for f in range(ref.shape[0]):
        noise = base.clone()
        noise[int(cu[f]) : int(cu[f + 1])] += 1.0
        expect = [i == f for i in range(ref.shape[0])]
        assert moved(_per_frame_losses(policy, batch, noise, t)) == expect, f"noise of frame {f}"
        tau = t.clone()
        tau[f] = 0.9
        assert moved(_per_frame_losses(policy, batch, base, tau)) == expect, f"τ of frame {f}"

    action_dim = group["group_items"][0][ACTION].shape[-1]
    noise = base.clone()
    noise[:, action_dim:] += 5.0
    assert torch.equal(_per_frame_losses(policy, batch, noise, t), ref), "padded dims leak"


def test_register_all_frames_supervised_and_grads(synthetic_group: Callable[..., Any]) -> None:
    policy, batch, _ = _policy_and_batch(synthetic_group)
    per_frame = _per_frame_flow_loss(policy, batch)
    assert per_frame.shape[0] == 3
    assert torch.isfinite(per_frame).all() and bool((per_frame > 0).all())

    with torch.no_grad():
        policy.model.state_adaln_proj.weight.zero_()
    loss, _ = policy.forward(batch)
    loss.backward()
    for name, p in (
        ("registers", policy.model.vlm_with_expert.action_expert.registers),
        ("state_adaln_proj", policy.model.state_adaln_proj.weight),
    ):
        assert p.grad is not None and float(p.grad.abs().sum()) > 0, f"{name} gets no gradient"


def test_dropped_state_is_the_learned_null(synthetic_group: Callable[..., Any]) -> None:
    """``state_dropout_prob=0.2`` in every recipe: a state-less sample's cond is exactly
    ``state_adaln_proj.bias``, while a present state moves it."""
    policy, _, group = _policy_and_batch(synthetic_group)
    model = policy.model
    with torch.no_grad():
        model.state_adaln_proj.bias.normal_()
    member = group["group_items"][-1]
    emb, state = member["state_mlp_embodiment"], member[OBS_STATE].to(DEVICE)
    cond = model._state_adaln_cond([emb, None], [state, None])
    bias = model.state_adaln_proj.bias.float()
    assert torch.equal(cond[1].float(), bias)
    assert not torch.allclose(cond[0].float(), bias)


def test_compiled_inference_matches_eager(synthetic_group: Callable[..., Any]) -> None:
    """``compile_inference`` (compiled expert graph + prefill graph) predicts
    the eager chunks across the cache fill and eviction, also under ``torch.inference_mode``
    (as LeRobot calls it), and a returned chunk survives the next replay. A cache built under
    ``torch.inference_mode`` can be reset outside it."""
    policy, _, group = _policy_and_batch(synthetic_group)
    policy.eval()
    batches = [_member_infer_batch(m) for m in group["group_items"]] * 2

    def episode() -> list[torch.Tensor]:
        policy.reset_episode()
        out = []
        for i, batch in enumerate(batches):
            torch.manual_seed(i)
            out.append(policy.predict_action_chunk(batch))
        return out

    with torch.inference_mode():  # builds the cache; later episodes reset it outside
        eager = [a.clone() for a in episode()]
    policy.compile_inference()
    episode()  # compiles the expert and captures the graphs
    compiled = episode()
    held = compiled[0].clone()
    episode()
    assert torch.equal(compiled[0], held), "a returned chunk was overwritten by a replay"
    with torch.inference_mode():
        compiled_inference_mode = episode()
    for i, (a, b) in enumerate(zip(eager * 2, compiled + compiled_inference_mode, strict=True)):
        scale = a.abs().max().clamp_min(1.0).item()
        diff = (a - b).abs().max().item()
        assert diff < 2e-2 * scale, f"predict {i}: compiled vs eager max|Δ|={diff} (scale {scale})"
