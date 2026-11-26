"""Vision-language (VL) co-training data: the FineVision / LLaVA-OneVision / VL spatial
datasets, and ``VLSampleAdapter``, which turns their QA samples into the sample format of
the robot data (tokens, vision inputs, labels).
"""

from __future__ import annotations

import bisect
import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS
from PIL import Image
from torch.utils.data import Dataset

from reki.data.robotics.tokenization import count_vision_tokens
from reki.data.text.cached_arrow_dataset import CachedArrowDataset
from reki.data.text.finevision import CachedArrowFineVision
from reki.data.text.llavaov_onefive import LLava15OV
from reki.data.text.vl_spatial_parquet import VLSpatialParquet
from reki.paths import finevision_path, llava_ext_path
from reki.policy.constants import IGNORE_INDEX

logger = logging.getLogger(__name__)

FINEVISION_FOR_ROBOTICS = {
    "LLaVA_Instruct_150K",
    "a_okvqa",
    "alfworldgpt",
    "allava_laion",
    "allava_vflan",
    "aokvqa",
    "art",
    "cambrian(filtered)_processed",
    "clevr",
    "clevr_math",
    "coco_colors",
    "cocoqa",
    "cocotext",
    "densefusion_1m",
    "drivelm",
    "est_vqa",
    "google_landmarks",
    "image_textualization(filtered)",
    "indoor_qa",
    "laion_gpt4v",
    "lnqa",
    "localized_narratives",
    "lrv_normal(filtered)",
    "lvis_instruct4v",
    "mimic_cgd",
    "mmevol",
    "nlvr2",
    "objects365_qa",
    "ocrvqa",
    "oodvqa",
    "scienceqa",
    "scienceqa(nona_context)",
    "sharegpt4o",
    "sharegpt4v(coco)",
    "sharegpt4v(knowledge)",
    "sharegpt4v(llava)",
    "sharegpt4v(sam)",
    "sketchyvqa",
    "spatialsense",
    "spot_the_diff",
    "st_vqa",
    "super_clevr(mathv360k)",
    "tallyqa",
    "textcaps",
    "textocr(gpt4v)",
    "textvqa",
    "vision_flan(filtered)",
    "visual7w",
    "vqav2",
    "vsr",
    "wildvision",
}
LLAVA_EXT_FOR_ROBOTICS = {
    "gqa",
    "llava_cot_100k",
    "llava_wild",
    "llrv_gpt4v",
    "sherlock",
    "svit",
    "viquae",
    "vision_oritented",
    "visual_chat",
}


VL_SPATIAL_CONFIGS: dict[str, tuple[str, ...]] = {
    "robopoint": ("robopoint_object_ref", "robopoint_region_ref"),
    "refspatial": (
        "refspatial_2d_choice",
        "refspatial_2d_reasoning",
        "refspatial_3d_choice",
        "refspatial_3d_multi_view",
        "refspatial_3d_reasoning",
        "refspatial_3d_vacant",
        "refspatial_3d_visual_choice",
    ),
    "pixmo": ("pixmo_points",),
}


def llava_ext_configs() -> list[str]:
    """The LLaVA-OV config dirs of ``LLAVA_EXT_FOR_ROBOTICS``; ``svit`` ships as 10 parts."""
    return [
        c
        for stem in sorted(LLAVA_EXT_FOR_ROBOTICS)
        for c in ([f"svit-part-{i:02d}-of-10" for i in range(10)] if stem == "svit" else [stem])
    ]


VL_SUBSET_MARKER = "SUBSET.json"
"""Written by a ``--subset`` build of ``scripts/data/pretrain_data.py`` into the FineVision /
LLaVA-OV root: the configs it holds (a JSON list)."""


def _subset(root: str, configs: list[str]) -> list[str] | None:
    """The configs of ``configs`` that a subset root holds; None: not a subset root."""
    marker = Path(root) / VL_SUBSET_MARKER
    if not marker.is_file():
        return None
    held = set(json.loads(marker.read_text()))
    return [c for c in configs if c in held]


def _build_finevision_inner(max_num_qa_pairs: int) -> list[CachedArrowDataset]:
    """The FineVision + LLaVA-OV inner datasets of ``finevision_for_robotics``, in sorted order.
    A subset root (``VL_SUBSET_MARKER``) gives only its configs."""
    fv_configs = sorted(FINEVISION_FOR_ROBOTICS)
    llava_configs = llava_ext_configs()
    fv_subset = _subset(finevision_path(), fv_configs)
    llava_subset = _subset(llava_ext_path(), llava_configs)
    if fv_subset is not None or llava_subset is not None:
        logger.warning(
            "finevision_for_robotics: SUBSET root (%s), %d of %d FineVision and %d of %d "
            "LLaVA-OV configs; not the training mixture",
            VL_SUBSET_MARKER,
            len(fv_configs if fv_subset is None else fv_subset),
            len(fv_configs),
            len(llava_configs if llava_subset is None else llava_subset),
            len(llava_configs),
        )
        fv_configs = fv_configs if fv_subset is None else fv_subset
        llava_configs = llava_configs if llava_subset is None else llava_subset
    logger.info(
        "Loading finevision_for_robotics: %d FineVision + %d LLaVA-OV configs",
        len(fv_configs),
        len(llava_configs),
    )
    with ThreadPoolExecutor() as executor:
        inner: list[CachedArrowDataset] = list(
            executor.map(lambda c: CachedArrowFineVision(c, max_num_qa_pairs), fv_configs)
        )
        inner.extend(executor.map(lambda c: LLava15OV(c, max_num_qa_pairs), llava_configs))
    return inner


def _resolve_vl_inner(token: str, max_num_qa_pairs: int) -> list[CachedArrowDataset]:
    """Map one VL spec token to its inner dataset(s)."""
    if token == "finevision_for_robotics":
        return _build_finevision_inner(max_num_qa_pairs)
    if token in VL_SPATIAL_CONFIGS:
        return [VLSpatialParquet(c, max_num_qa_pairs) for c in VL_SPATIAL_CONFIGS[token]]
    raise ValueError(
        f"Unknown VL dataset token: {token!r}. Known: "
        f"{['finevision_for_robotics', *VL_SPATIAL_CONFIGS]}"
    )


def make_vl_dataset(
    vl_datasets: str | tuple[str, ...] | list[str],
    preprocessor: Any,
    max_num_qa_pairs: int = 1,
) -> VLSampleAdapter:
    """Build a VLSampleAdapter over a comma-separated VL dataset spec.

    Tokens (de-duplicated, order-preserving):
      - ``finevision_for_robotics`` — curated FineVision + LLaVA-OV subset (see
        ``docs/data.md``);
      - ``robopoint``, ``refspatial``, ``pixmo`` — VL spatial datasets loaded from
        the FineVision-format Arrow or parquet at ``reki.paths.vl_spatial_parquet_path()``,
        one inner dataset per config of ``VL_SPATIAL_CONFIGS``.

    All inner datasets are concatenated into one VLSampleAdapter; sampling
    exposure within the block is ∝ row count (no per-source caps).
    """
    tokens = vl_datasets.split(",") if isinstance(vl_datasets, str) else list(vl_datasets)
    seen: set[str] = set()
    inner_datasets: list[Any] = []
    for token in tokens:
        token = token.strip()
        if not token or token in seen:
            continue
        seen.add(token)
        inner_datasets.extend(_resolve_vl_inner(token, max_num_qa_pairs))

    logger.info("VL dataset (%s): %d inner sub-datasets loaded", vl_datasets, len(inner_datasets))
    return VLSampleAdapter(
        inner_datasets=inner_datasets,
        preprocessor=preprocessor,
    )


class VLSampleAdapter(Dataset):
    """Wraps CachedArrowFineVision / LLava15OV datasets into the REKI sample format.

    Delegates tokenization + image preprocessing to ``Qwen3_5Preprocessor``; each
    sample returns ``input_ids``, ``labels`` and the ``vision_inputs`` dict consumed by
    ``Qwen3_5VLMWithExpert``.
    """

    def __init__(
        self,
        inner_datasets: list[Any],
        preprocessor: Any,
    ) -> None:
        self._datasets = inner_datasets
        self._preprocessor = preprocessor

        self._cum_sizes: list[int] = []
        total = 0
        for ds in self._datasets:
            total += len(ds)
            self._cum_sizes.append(total)

    def __len__(self) -> int:
        return self._cum_sizes[-1] if self._cum_sizes else 0

    _MAX_RETRIES = 5

    def __getitem__(self, idx: int) -> dict[str, Any]:
        for attempt in range(self._MAX_RETRIES):
            try:
                return self._getitem_inner(idx)
            except OSError as e:
                logger.warning(
                    "VLSampleAdapter: image read error at idx=%d (attempt %d/%d): %s",
                    idx,
                    attempt + 1,
                    self._MAX_RETRIES,
                    e,
                )
                idx = random.randint(0, len(self) - 1)
            except Exception as e:
                ds, local = self._resolve_index(idx)
                raise RuntimeError(
                    f"VL sample idx={idx} (dataset={ds.dataset_name}, local={local}) failed: "
                    f"{type(e).__name__}: {e}"
                ) from e
        raise RuntimeError(f"All {self._MAX_RETRIES} VL retries hit image read errors")

    def _resolve_index(self, idx: int) -> tuple[Any, int]:
        """Map a global index to (dataset, local_index)."""
        ds_idx = bisect.bisect_right(self._cum_sizes, idx)
        offset = self._cum_sizes[ds_idx - 1] if ds_idx > 0 else 0
        local_idx = idx - offset
        return self._datasets[ds_idx], local_idx

    def _getitem_inner(self, idx: int) -> dict[str, Any]:
        ds, real_idx = self._resolve_index(idx)
        raw_item = ds.ds[real_idx]

        qa_pairs: list[tuple[int, str]] = list(ds.prepare_for_formatter(raw_item))

        pil_images = raw_item.get(ds.img_key, [])
        if pil_images is None:
            pil_images = []
        if not isinstance(pil_images, list):
            pil_images = [pil_images]
        pil_images = [
            img.convert("RGBA").convert("RGB") if isinstance(img, Image.Image) else img
            for img in pil_images
            if img is not None
        ]
        if not pil_images:
            raise ValueError("Sample has no images")
        input_ids, vision_inputs, labels = self._tokenize_with_labels(qa_pairs, pil_images)

        return {
            OBS_LANGUAGE_TOKENS: input_ids,
            "vision_inputs": vision_inputs,
            "n_image_tokens": count_vision_tokens(vision_inputs),
            "labels": labels,
            "camera_indices": torch.tensor([], dtype=torch.long),
        }

    def _tokenize_with_labels(
        self,
        qa_pairs: list[tuple[int, str]],
        pil_images: list,
    ) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor]:
        """Tokenize a QA conversation; the labels supervise only the assistant answers.

        Each assistant turn is found by its multi-token header ``assistant_header_ids`` (as in
        ``FrameTokenizer``); the tokens after it, up to and including ``<|im_end|>``, are
        supervised. The user turn (text and image-pad tokens) and the header are
        ``IGNORE_INDEX``. On Qwen3.5, ``asst_start_tokens`` is only a ``"\\n"`` separator, so one
        token cannot mark the turn. Returns (input_ids, vision_inputs, labels).
        """
        prep = self._preprocessor

        messages: list[dict[str, Any]] = []
        first_user = True
        for speaker_id, text in qa_pairs:
            role = "user" if speaker_id == 1 else "assistant"
            content: list[dict[str, Any]] = []
            if role == "user" and first_user and pil_images:
                content.extend({"type": "image", "image": img} for img in pil_images)
                first_user = False
            content.append({"type": "text", "text": text})
            messages.append({"role": role, "content": content})

        sample = prep.preprocess(messages=messages)
        input_ids = sample["input_ids"]
        vision_inputs = sample["vision_inputs"]

        asst_end_ids = torch.tensor(prep.asst_end_tokens, dtype=input_ids.dtype)
        if prep.asst_end_tokens[0] not in input_ids[-2:].tolist():
            input_ids = torch.cat([input_ids, asst_end_ids])
            mm_ids = vision_inputs.get("mm_token_type_ids")
            if mm_ids is not None:
                pad = torch.zeros((*mm_ids.shape[:-1], len(asst_end_ids)), dtype=mm_ids.dtype)
                vision_inputs["mm_token_type_ids"] = torch.cat([mm_ids, pad], dim=-1)

        header = list(prep.assistant_header_ids)
        if not header:
            raise ValueError(
                "preprocessor.assistant_header_ids is empty; cannot locate assistant "
                "turns for VL label masking (the Qwen3.5 preprocessor defines one)."
            )
        asst_end_id = prep.asst_end_tokens[0]
        ids_list = input_ids.tolist()
        n, h = len(ids_list), len(header)
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        i = 0
        while i <= n - h:
            if ids_list[i : i + h] == header:
                j = i + h
                while j < n and ids_list[j] != asst_end_id:
                    j += 1
                end = j + 1 if j < n else n
                labels[i + h : end] = input_ids[i + h : end]
                i = end
            else:
                i += 1

        return input_ids, vision_inputs, labels
