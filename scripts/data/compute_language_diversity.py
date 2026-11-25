"""Language diversity of each dataset: the effective rank of the SigLIP text embeddings of the
task texts that training reads, weighted by frames.

The text of a frame is the task of the episode in the subtask lookup, else the frame's task in
``meta/tasks.parquet``. Frames of the episodes in ``meta/dropped_episodes.json`` do not count.
The scores are merged into the config's ``language_diversity_path``; the sampling weight of a
dataset is multiplied by ``(score / median score) ** language_alpha``.

    pixi run python scripts/data/compute_language_diversity.py --config configs/cpt.yaml
    pixi run python scripts/data/compute_language_diversity.py --config configs/cpt.yaml \
        --datasets droid galaxea
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from reki.data.robotics.lerobot_io import resolve_lerobot_dirs
from reki.data.robotics.prompt import clean_task_text, normalize_task
from reki.data.robotics.registry import spec_names
from reki.data.robotics.subtasks import load_subtask_lookup
from reki.policy.train_config import TrainConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def leaf_text_frames(leaf: Path, lookup_filename: str) -> Counter[str]:
    """Frames per task text of one leaf, as training reads them."""
    tasks = pd.read_parquet(leaf / "meta" / "tasks.parquet")
    texts = tasks["task"] if "task" in tasks.columns else tasks.index.to_series()
    task_text = {int(i): normalize_task(str(t)) for i, t in zip(tasks["task_index"], texts)}
    override = load_subtask_lookup(leaf, lookup_filename).tasks or {}
    dropped_path = leaf / "meta" / "dropped_episodes.json"
    dropped = set(json.loads(dropped_path.read_text())) if dropped_path.exists() else set()
    counts: Counter[str] = Counter()
    for f in sorted((leaf / "data").rglob("*.parquet")):
        t = pq.read_table(f, columns=["episode_index", "task_index"])
        pairs = np.stack([t["episode_index"].to_numpy(), t["task_index"].to_numpy()], 1)
        uniq, n = np.unique(pairs, axis=0, return_counts=True)
        for (ep, ti), k in zip(uniq.tolist(), n.tolist()):
            if ep not in dropped:
                counts[clean_task_text(override.get(ep) or task_text[ti])] += k
    return counts


def embed(texts: list[str], model_name: str, device: str, batch: int = 256) -> np.ndarray:
    """SigLIP text embeddings ``[N, D]``."""
    from transformers import AutoModel, PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)  # no sentencepiece needed
    text_model = AutoModel.from_pretrained(model_name).to(device).eval().text_model
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            tokens = tokenizer(
                texts[i : i + batch], padding="max_length", truncation=True, return_tensors="pt"
            ).to(device)
            out.append(text_model(**tokens).pooler_output.float().cpu().numpy())
    return np.concatenate(out)


def effective_rank(embeddings: np.ndarray, weights: np.ndarray) -> float:
    """``2 ** H(s² / Σs²)``, ``s`` the singular values of the L2-normalized rows scaled by
    ``sqrt(weight)`` (not centered: two texts give a rank above 1)."""
    rows = embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12, None)
    s = np.linalg.svd(rows * np.sqrt(weights / weights.sum())[:, None], compute_uv=False)
    p = s**2 / (s**2).sum()
    p = p[p > 0]
    return float(2 ** -(p * np.log2(p)).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--datasets", nargs="*", help="default: the datasets of the config")
    parser.add_argument("--model", default="google/siglip-base-patch16-224")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    cfg = TrainConfig.from_config_and_overrides(args.config, {})
    if not cfg.language_diversity_path:
        raise ValueError(f"{args.config}: no language_diversity_path")
    names = args.datasets or spec_names(cfg.dataset)

    jobs = [(name, leaf) for name in names for leaf in resolve_lerobot_dirs(name)]
    with ThreadPoolExecutor(args.workers) as ex:
        leaf_counts = list(
            ex.map(lambda j: leaf_text_frames(j[1], cfg.subtask_lookup_filename), jobs)
        )
    per_dataset: dict[str, Counter[str]] = {name: Counter() for name in names}
    for (name, _), counts in zip(jobs, leaf_counts):
        per_dataset[name].update(counts)

    texts = sorted({t for c in per_dataset.values() for t in c})
    logger.info("embedding %d texts with %s", len(texts), args.model)
    emb = embed(texts, args.model, "cuda" if torch.cuda.is_available() else "cpu")
    row = {t: i for i, t in enumerate(texts)}
    scores = {}
    for name, counts in per_dataset.items():
        if not counts:
            raise ValueError(f"{name}: no kept frames")
        idx = np.array([row[t] for t in counts])
        scores[name] = effective_rank(emb[idx], np.array(list(counts.values()), np.float64))
        logger.info("%s: %.3f over %d texts", name, scores[name], len(counts))

    out = Path(cfg.language_diversity_path)
    merged = {**json.loads(out.read_text()), **scores} if out.exists() else scores
    merged = {k: round(v, 3) for k, v in sorted(merged.items(), key=lambda kv: kv[1])}
    out.write_text(json.dumps(merged, indent=2) + "\n")
    logger.info("wrote %d scores to %s", len(scores), out)


if __name__ == "__main__":
    main()
