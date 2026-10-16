"""Loading checkpoint weights into a ``RekiModel``: resolve a local dir or HF model id, read the
tensors, overlay the EMA, seed new state-MLP banks."""

from __future__ import annotations

import os
from pathlib import Path

import safetensors.torch
import torch

from reki.data.robotics.registry import DATASETS, mirrored_name


def _group_keys(keys: list[str], reference: list[str] | set[str]) -> list[str]:
    """Shorten ``keys``, a subset of ``reference``, with ``.*`` wildcards for display.

    A node becomes ``<prefix>.* (<n>)`` only when every leaf under it is in ``keys``, so a
    wildcard never hides a ``reference`` key that is not in ``keys``."""
    root: dict[str, dict] = {}
    for k in reference:
        node = root
        for part in k.split("."):
            node = node.setdefault(part, {})

    listed = set(keys)

    def all_listed(node: dict[str, dict], prefix: list[str]) -> bool:
        if not node:
            return ".".join(prefix) in listed
        return all(all_listed(c, prefix + [n]) for n, c in node.items())

    def count_leaves(node: dict[str, dict]) -> int:
        if not node:
            return 1
        return sum(count_leaves(c) for c in node.values())

    def single_path(node: dict[str, dict], prefix: list[str]) -> list[str]:
        cur, parts = node, list(prefix)
        while cur:
            name, child = next(iter(cur.items()))
            parts.append(name)
            cur = child
        return parts

    out: list[str] = []

    def emit(node: dict[str, dict], prefix: list[str]) -> None:
        if not node:
            key = ".".join(prefix)
            if key in listed:
                out.append(key)
            return
        if prefix and all_listed(node, prefix):
            n = count_leaves(node)
            if n == 1:
                out.append(".".join(single_path(node, prefix)))
            else:
                out.append(f"{'.'.join(prefix)}.* ({n})")
            return
        for n, c in sorted(node.items()):
            emit(c, prefix + [n])

    emit(root, [])
    return out


_MIRROR_SEED_ROOT: str = "model.vlm_with_expert.state_mlps."
_MIRROR_SEED_PREFIXES: tuple[tuple[str, str], ...] = tuple(
    (f"{_MIRROR_SEED_ROOT}{emb}.", f"{_MIRROR_SEED_ROOT}{mirrored_name(emb)}.")
    for emb in dict.fromkeys(
        spec.embodiment for spec in DATASETS.values() if spec.mirror and spec.embodiment is not None
    )
)


def seed_mirrored_state_mlp(state_dict: dict, model_keys: set[str]) -> int:
    """Copy the trained per-embodiment banks onto their ``<emb>_mirrored`` banks, for a
    checkpoint trained without mirroring; return the number of tensors copied. The normalized
    mirrored state is a signed permutation of the plain one, so the copy is a warm start. Call it
    before the load. A pair is skipped when the model has no mirrored bank, the checkpoint
    already has one, or the checkpoint has no base bank.
    """
    total = 0
    for src, dst in _MIRROR_SEED_PREFIXES:
        if not any(k.startswith(dst) for k in model_keys):
            continue
        if any(k.startswith(dst) for k in state_dict):
            continue
        seeded = {
            dst + k[len(src) :]: v.clone() for k, v in state_dict.items() if k.startswith(src)
        }
        state_dict.update(seeded)
        total += len(seeded)
    return total


_PROPRIO_PREFIXES: tuple[str, ...] = (
    "model.vlm_with_expert.state_mlps.",
    "model.state_adaln_proj.",
)


def check_proprio_shapes(state_dict: dict, model_state: dict) -> None:
    """Raise when a state MLP of the checkpoint has another shape than the model's: the state
    format kept its name but changed its dim."""
    resized = sorted(
        k
        for k, v in state_dict.items()
        if k.startswith(_PROPRIO_PREFIXES) and k in model_state and model_state[k].shape != v.shape
    )
    if resized:
        shapes = ", ".join(
            f"{k}: {tuple(state_dict[k].shape)} != {tuple(model_state[k].shape)}" for k in resized
        )
        raise ValueError(
            f"state MLP shapes differ from the checkpoint ({shapes}). Keep the checkpoint's dim "
            "for this state format, or use a new state format key for a different state layout."
        )


def resolve_checkpoint(
    path_or_hf_id: str | Path, include_ema: bool = False, revision: str | None = None
) -> Path:
    """Return a local checkpoint directory as is. Otherwise download the HF model repo
    ``org/name`` at ``revision`` into the HF cache (``ema.safetensors`` only with
    ``include_ema``). A string that looks like a path (or whose first component is a local
    directory) and does not exist raises ``FileNotFoundError``."""
    p = Path(path_or_hf_id)
    if p.is_dir():
        return p
    s = str(path_or_hf_id)
    if (
        p.exists()
        or s.startswith(("/", ".", "~"))
        or s.count("/") != 1
        or Path(p.parts[0]).is_dir()
    ):
        raise FileNotFoundError(f"checkpoint directory not found: {s}")
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        return Path(
            snapshot_download(
                repo_id=s,
                revision=revision,
                ignore_patterns=None if include_ema else ["ema.safetensors"],
            )
        )
    except RepositoryNotFoundError as e:
        raise FileNotFoundError(
            f"{s!r} is not a local folder and is not a model id on https://huggingface.co/models. "
            "If it is a private repository, log in with `hf auth login`."
        ) from e


def load_checkpoint_tensors(path: str | Path) -> dict[str, torch.Tensor]:
    """A checkpoint's ``.safetensors`` file."""
    return safetensors.torch.load_file(str(path))


def load_pretrained_weights(
    policy: torch.nn.Module,
    pretrained_path: str | Path,
    use_ema: bool = False,
) -> None:
    """Load model weights from a checkpoint directory or HF repo id (``resolve_checkpoint``).

    State-MLP entries that the model does not use are skipped. This is normal when finetuning
    on a data subset (e.g. LIBERO only). Missing keys raise, except proprio keys
    (``_PROPRIO_PREFIXES``): a warm start onto a new embodiment adds a new state MLP.

    With ``use_ema=True``, ``ema.safetensors`` is loaded over ``model.safetensors``. The
    EMA has only the trainable params; the frozen params come from model.safetensors. The EMA
    file must exist.
    """
    from reki.utils.dist_utils import print_main

    ckpt_dir = resolve_checkpoint(pretrained_path, include_ema=use_ema)
    ckpt_file = str(ckpt_dir / "model.safetensors")
    print_main(f"Loading pretrained weights from {ckpt_file}")
    state_dict = load_checkpoint_tensors(ckpt_file)
    if use_ema:
        ema_file = str(ckpt_dir / "ema.safetensors")
        assert os.path.exists(ema_file), f"use_ema=True but {ema_file} not found"
        ema_state = load_checkpoint_tensors(ema_file)
        overlay = {k: v for k, v in ema_state.items() if k in state_dict}
        skipped = sorted(set(ema_state.keys()) - set(state_dict.keys()))
        state_dict.update(overlay)
        print_main(f"  Overlaid {len(overlay)}/{len(ema_state)} EMA keys from {ema_file}")
        if skipped:
            print_main(
                f"  Skipped EMA keys (not in model.safetensors): "
                f"{_group_keys(skipped, list(ema_state.keys()))}"
            )
    model_state = policy.state_dict()
    model_keys = set(model_state)
    check_proprio_shapes(state_dict, model_state)
    seeded = seed_mirrored_state_mlp(state_dict, model_keys)
    if seeded:
        print_main(f"  Seeded {seeded} mirrored state-MLP tensors from their base banks")

    missing, unexpected = policy.load_state_dict(state_dict, strict=False)

    # Tied embeddings: a checkpoint that stores lm_head may omit embed_tokens.
    embed_key = "model.vlm_with_expert.vlm.model.language_model.embed_tokens.weight"
    if "model.vlm_with_expert.vlm.lm_head.weight" in state_dict:
        missing = [k for k in missing if k != embed_key]

    tolerated = [k for k in unexpected if k.startswith("model.vlm_with_expert.state_mlps.")]
    unexpected = [k for k in unexpected if k not in tolerated]
    if unexpected:
        raise ValueError(f"Unexpected keys loading pretrained_path={pretrained_path}: {unexpected}")
    non_proprio = [k for k in missing if not k.startswith(_PROPRIO_PREFIXES)]
    if non_proprio:
        raise ValueError(f"Missing keys loading pretrained_path={pretrained_path}: {non_proprio}")
    loaded = len(state_dict) - len(tolerated)
    print_main(f"  Loaded {loaded}/{len(state_dict)} keys ({len(missing)} missing: new proprio)")
    if tolerated:
        print_main(
            f"  Skipped (unused by this run): {_group_keys(tolerated, list(state_dict.keys()))}"
        )
    if missing:
        print_main(f"  New, random init: {_group_keys(missing, list(model_keys))}")
