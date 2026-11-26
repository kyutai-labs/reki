"""MANO fit of 21 hand keypoints (otaheri-MANO order) → 45-dim finger axis-angle and wrist pose
(GPU, Adam).

Each frame has its own MANO parameters (pose, shape, global orient, translation); the frames
are fitted in batches of ``CHUNK``, packed by ``chunks``. A frame's result depends on its batch:
the loss is normalized over the whole batch, and padding rows round differently. Two runs on one
GPU type and software stack give identical results; another GPU, torch or TF32 setting changes
the low bits, which can also move a frame across the ``fit_err`` threshold. Needs the ``mano``
package (otaheri) and the MANO model files (registration at mano.is.tue.mpg.de, not
redistributable).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

N_ITERS = 250
LR = 0.04
BETA_REG = 0.01
CHUNK = 16384


class FastMano:
    """MANO forward that skins only the 5 fingertip vertices + the 16 joints (21 points);
    matches ``model(...).joints`` to ~1e-4 mm. ``hand_pose`` is 45-dim axis-angle."""

    def __init__(self, model: Any, device: torch.device) -> None:
        self.dev = device
        self.parents = model.parents
        self.J_regressor = model.J_regressor
        self.v_template = model.v_template
        self.shapedirs = model.shapedirs
        self.pose_mean = model.pose_mean
        self.dtype = model.dtype
        tip_ids = torch.tensor(list(model.tip_ids.values()), device=device)
        self.vt_tip = self.v_template.index_select(0, tip_ids)
        self.sd_tip = self.shapedirs.index_select(0, tip_ids)
        posedirs = model.posedirs.view(model.posedirs.shape[0], 778, 3).index_select(1, tip_ids)
        self.pd_tip = posedirs.reshape(model.posedirs.shape[0], 15).T
        self.lw_tip = model.lbs_weights.index_select(0, tip_ids)

    def joints(
        self,
        betas: torch.Tensor,
        global_orient: torch.Tensor,
        hand_pose: torch.Tensor,
        transl: torch.Tensor,
    ) -> torch.Tensor:
        from mano.lbs import (  # type: ignore[import-not-found]
            batch_rigid_transform,
            batch_rodrigues,
            blend_shapes,
            vertices2joints,
        )

        B = betas.shape[0]
        full_pose = torch.cat([global_orient, hand_pose], dim=1) + self.pose_mean
        v_shaped = self.v_template + blend_shapes(betas, self.shapedirs)
        J = vertices2joints(self.J_regressor, v_shaped)
        rot = batch_rodrigues(full_pose.view(-1, 3)).view(B, -1, 3, 3)
        J_t, A = batch_rigid_transform(rot, J, self.parents, dtype=self.dtype)
        vt = self.vt_tip + torch.einsum("bl,mkl->bmk", betas, self.sd_tip)
        pose_feature = (rot[:, 1:, :, :] - torch.eye(3, device=self.dev)).reshape(B, -1)
        v_posed = vt + (pose_feature @ self.pd_tip.T).view(B, 5, 3)
        T = (self.lw_tip.unsqueeze(0).expand(B, -1, -1) @ A.view(B, 16, 16)).view(B, 5, 4, 4)
        vh = torch.cat([v_posed, torch.ones(B, 5, 1, device=self.dev)], dim=2)
        tips = (T[:, :, :3, :] * vh.unsqueeze(2)).sum(-1)
        return torch.cat([J_t, tips], dim=1) + transl.unsqueeze(1)


def _fit_chunk(keypoints: torch.Tensor, fast: FastMano, device: torch.device) -> dict:
    """``[N, 21, 3]`` keypoints (non-finite = missing point) → per-frame MANO parameters."""
    N = keypoints.shape[0]
    kp = keypoints.to(device)
    kp = torch.where(kp.abs() < 1e8, kp, torch.full_like(kp, float("nan")))
    point_valid = torch.isfinite(kp).all(dim=-1)
    targets = torch.nan_to_num(kp, nan=0.0)

    theta = torch.zeros(N, 45, device=device, requires_grad=True)
    beta = torch.zeros(N, 10, device=device, requires_grad=True)
    orient = torch.zeros(N, 3, device=device, requires_grad=True)
    transl = targets[:, 0, :].detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([theta, beta, orient, transl], lr=LR)
    for _ in range(N_ITERS):
        pred = fast.joints(beta, orient, theta, transl)
        err = ((pred - targets) ** 2).sum(-1) * point_valid.float()
        loss = err.sum() / point_valid.sum().clamp(min=1) + BETA_REG * (beta**2).sum() / N
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    params = {"betas": beta, "global_orient": orient, "hand_pose": theta, "transl": transl}
    return {k: v.detach() for k, v in params.items()} | {"targets": targets, "valid": point_valid}


def _readout(model: Any, chunk: dict, sl: slice) -> dict[str, np.ndarray]:
    """Finger pose, wrist pose (the MANO wrist joint and global orient) and mean per-point fit
    error (m) of the frames ``sl`` of a fitted chunk."""
    with torch.no_grad():
        out = model(
            **{k: chunk[k][sl] for k in ("betas", "global_orient", "hand_pose", "transl")},
            return_verts=True,
            return_tips=True,
            return_full_pose=True,
        )
        valid = chunk["valid"][sl]
        per_point = torch.sqrt(((out.joints - chunk["targets"][sl]) ** 2).sum(-1))
        fit_err = (per_point * valid).sum(1) / valid.sum(1).clamp(min=1)
    return {
        "finger_aa": out.full_pose[:, 3:48].cpu().numpy(),
        "wrist_t": out.joints[:, 0].cpu().numpy(),
        "wrist_aa": chunk["global_orient"][sl].cpu().numpy(),
        "fit_err": fit_err.cpu().numpy(),
    }


def chunks(lengths: list[int]) -> list[list[tuple[int, int]]]:
    """Episodes ``(start, length)`` packed whole into chunks of at most ``CHUNK - 1`` frames, in
    order; an episode longer than that is split into chunks of its own. The last row of a
    chunk is always padding: a frame there rounds differently."""
    cap = CHUNK - 1
    out: list[list[tuple[int, int]]] = []
    used = cap
    start = 0
    for n in lengths:
        for s in range(0, n, cap):
            piece = min(cap, n - s)
            if used + piece > cap or n > cap:
                out.append([])
                used = 0
            out[-1].append((start + s, piece))
            used += piece
        if n > cap:
            used = cap
        start += n
    return out


def fit_hand(
    keypoints: np.ndarray, lengths: list[int], is_right: bool, mano_dir: Path
) -> dict[str, np.ndarray]:
    """MANO fit of the ``[T, 21, 3]`` keypoints of one hand over episodes of ``lengths``.

    Episodes are packed whole into zero-padded batches of ``CHUNK`` frames (``chunks``) and
    read out one episode at a time.
    """
    import mano  # type: ignore[import-not-found]

    if not torch.cuda.is_available():
        raise RuntimeError("the MANO fit needs a GPU")
    if sum(lengths) != len(keypoints):
        raise ValueError(f"{len(keypoints)} frames, episodes hold {sum(lengths)}")
    device = torch.device("cuda")
    model = mano.load(
        model_path=str(mano_dir),
        is_rhand=is_right,
        num_pca_comps=45,
        batch_size=CHUNK,
        flat_hand_mean=False,
    ).to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    fast = FastMano(model, device)
    parts: list[dict[str, np.ndarray]] = []
    for pieces in chunks(lengths):
        kp = np.concatenate([keypoints[s : s + n] for s, n in pieces])
        kp = np.concatenate([kp, np.zeros((CHUNK - len(kp), 21, 3), kp.dtype)])
        fitted = _fit_chunk(torch.as_tensor(kp, dtype=torch.float32), fast, device)
        offset = 0
        for _, n in pieces:
            parts.append(_readout(model, fitted, slice(offset, offset + n)))
            offset += n
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
