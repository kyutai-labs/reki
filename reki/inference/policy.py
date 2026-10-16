"""``RekiPolicy``: the public inference API (``from reki import RekiPolicy``).

.. code-block:: python

    policy = RekiPolicy.from_pretrained("kyutai/reki-2b-libero", device="cuda")
    policy.reset()
    actions = policy.predict(images={...}, state=state, task="put the bowl on the plate")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from reki.inference.features import EnvObs, build_inference_sample, decode_action, model_state
from reki.inference.spec import InferenceSpec, Stats, inference_spec_for, select_stats
from reki.policy.configuration import RekiConfig
from reki.policy.modeling import RekiModel
from reki.policy.train_config import TrainConfig
from reki.vlm.qwen3_5.preprocessor import Qwen3_5Preprocessor


def build_policy(train_cfg: TrainConfig, device: str) -> RekiModel:
    """The model of a training config, with a random VLM: load the full checkpoint after. Only
    the VLM config is fetched, not its weights."""
    return RekiModel(RekiConfig.from_train_config(train_cfg, device=device), load_vlm_weights=False)


def build_eval_preprocessor(train_cfg: TrainConfig) -> Qwen3_5Preprocessor:
    """The preprocessor of a training config, as training builds it, with the token cache on.
    A deployed policy tokenizes the same prompt at every step."""
    preprocessor = Qwen3_5Preprocessor.from_pretrained(
        train_cfg.vlm_model_id, image_size=train_cfg.image_size
    )
    preprocessor.token_cache = {}
    return preprocessor


class RekiPolicy:
    """A REKI checkpoint as a stateful policy.

    The model keeps the history of the episode in its register memory, so the calls must
    follow the episode: call ``reset()`` at the start of each episode, then ``predict`` once
    every ``n_action_steps`` environment steps, and execute the returned actions in between.
    Another cadence changes the frame spacing the model was trained on, and nothing detects it.

    Attributes: ``policy`` (the ``RekiModel``), ``spec`` (the ``InferenceSpec``), ``stats``
    (normalization stats), ``train_cfg`` (the training config of the checkpoint).
    """

    def __init__(
        self,
        policy: RekiModel,
        preprocessor: Any,
        train_cfg: TrainConfig,
        spec: InferenceSpec,
        stats: Stats,
        device: str,
    ) -> None:
        self.policy = policy
        self.preprocessor = preprocessor
        self.train_cfg = train_cfg
        self.spec = spec
        self.stats = stats
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        path_or_hf_id: str | Path,
        device: str = "cuda",
        dataset: str | None = None,
        use_ema: bool = True,
        compile: bool = True,
        revision: str | None = None,
    ) -> RekiPolicy:
        """Load a checkpoint.

        Args:
            path_or_hf_id: a local directory or a Hugging Face repo id. An exported checkpoint
                (``scripts/export_checkpoint.py``, all Hub checkpoints) has its inference spec;
                a raw training checkpoint needs ``dataset``.
            device: a CUDA device. The model needs flash-attn and FLA kernels; CPU is not
                supported.
            dataset: only for a raw training checkpoint: the registered dataset it was
                finetuned on. The spec and stats are then built from that dataset's ``meta/``
                under ``REKI_DATASETS_ROOT``.
            use_ema: overlay the EMA weights (``ema.safetensors``) when the checkpoint has
                them (``scripts/eval/rollout.py --use_ema``). A checkpoint without them (EMA
                already merged at export) loads ``model.safetensors`` either way.
            compile: on a CUDA device, compile the expert and replay the prefill as CUDA graphs
                (``RekiModel.compile_inference``). The first predict compiles.
            revision: the Hub revision (branch, tag or commit) of a Hugging Face repo id.
        """
        from reki.policy.checkpoint import load_pretrained_weights, resolve_checkpoint

        ckpt_dir = resolve_checkpoint(path_or_hf_id, include_ema=use_ema, revision=revision)
        train_cfg = TrainConfig.from_checkpoint(ckpt_dir)
        spec, stats = inference_spec_for(ckpt_dir, train_cfg, dataset)
        policy = build_policy(train_cfg, device).to(device)
        load_pretrained_weights(
            policy,
            str(ckpt_dir),
            use_ema=use_ema and (ckpt_dir / "ema.safetensors").exists(),
        )
        policy.eval()
        policy.config.chunk_size = spec.chunk_size
        if compile and torch.device(device).type == "cuda":
            policy.compile_inference()
        return cls(policy, build_eval_preprocessor(train_cfg), train_cfg, spec, stats, device)

    @property
    def n_action_steps(self) -> int:
        """Actions per ``predict`` call; call ``predict`` once every this many env steps."""
        return self.spec.n_action_steps

    def reset(self) -> None:
        """Start a new episode: clear the register memory."""
        self.policy.reset_episode()

    def _check_inputs(self, images: dict[str, np.ndarray], state: np.ndarray) -> None:
        want = list(self.spec.camera_keys)
        if sorted(images) != sorted(want):
            raise KeyError(f"images keys {sorted(images)} != expected cameras {want}")
        for k in want:
            img = images[k]
            if not isinstance(img, np.ndarray) or img.dtype != np.uint8:
                raise TypeError(
                    f"images[{k!r}] must be a uint8 numpy array, got {type(img).__name__} "
                    f"with dtype {getattr(img, 'dtype', None)}"
                )
            if img.ndim != 3 or img.shape[-1] != 3:
                raise ValueError(f"images[{k!r}] must be HxWx3, got shape {img.shape}")
        if not isinstance(state, np.ndarray) or not np.issubdtype(state.dtype, np.floating):
            raise TypeError(
                f"state must be a float numpy array, got {type(state).__name__} "
                f"with dtype {getattr(state, 'dtype', None)}"
            )
        if state.shape != (self.spec.state_dim,):
            raise ValueError(
                f"state shape {state.shape} != ({self.spec.state_dim},); layout: "
                f"{self.spec.state_layout}"
            )

    def predict(
        self,
        images: dict[str, np.ndarray],
        state: np.ndarray,
        task: str,
        stats_key: str | None = None,
    ) -> np.ndarray:
        """The next ``n_action_steps`` actions for one observation.

        Args:
            images: one image per camera, keyed by ``spec.camera_keys``: uint8 arrays of shape
                (H, W, 3), RGB, in the orientation of the training dataset. An image larger than
                ``image_size``² pixels is scaled down to that area, as in training.
            state: float array of shape (``spec.state_dim``,): the ``observation.state`` of the
                training dataset, as stored (layout: ``spec.state_layout``).
            task: the instruction as free text. It is normalized as in training
                (underscores to spaces, capitalized, ending with a period).
            stats_key: the task name, only for a checkpoint with per-task stats
                (``spec.per_task_stats``; the names are in ``spec.stats_keys``).

        Returns:
            float32 array of shape (``n_action_steps``, ``spec.action_dim``): unnormalized
            actions in the native units and rotation format of the training dataset.
        """
        self._check_inputs(images, state)
        state_stats, action_stats = select_stats(self.stats, self.spec.per_task_stats, stats_key)
        obs = EnvObs(
            images=[Image.fromarray(images[k]) for k in self.spec.camera_keys],
            state_raw=model_state(self.spec, state),
            task_text=task,
        )
        batch = build_inference_sample(
            obs,
            preprocessor=self.preprocessor,
            train_cfg=self.train_cfg,
            state_stats=state_stats,
            embodiment=self.spec.embodiment,
            control_mode=self.spec.control_mode,
            device=self.device,
            camera_indices=self.spec.camera_indices,
            robot_type=self.spec.robot_type,
            fps=self.spec.prompt_fps,
            action_dim=self.spec.action_dim,
            dataset=self.spec.dataset,
        )
        with torch.no_grad():
            chunk = self.policy.predict_action_chunk(batch)
        chunk = chunk[0, : self.spec.n_action_steps, : self.spec.action_dim].to(torch.float32).cpu()
        rot = self.spec.euler_rot_dims
        # The whole chunk in one call: every decode op is row-wise.
        return decode_action(
            chunk,
            action_stats,
            dataset=self.spec.dataset,
            euler_rot_dims=(rot[0], rot[1]) if rot is not None else None,
        ).astype(np.float32)
