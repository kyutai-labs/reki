"""Vision diversity of each dataset: the effective rank of the DINOv3 CLS embeddings of frames of
the primary camera that training reads.

The primary camera of a leaf is its first camera in the loader's ranking
(``cameras.select_top_cameras``). The video files of a dataset are shuffled (seed 0), and every
``--stride``-th frame of each file is taken, in that file order, up to ``--max-frames``. Frames of
the episodes in ``meta/dropped_episodes.json`` do not count. The scores are merged into the
config's ``vision_diversity_path``; the sampling weight of a dataset is multiplied by
``(score / median score) ** vision_alpha``.

Video decode dominates the cost: run it on one GPU with many CPUs.

    pixi run python scripts/data/compute_vision_diversity.py --config configs/cpt.yaml
    pixi run python scripts/data/compute_vision_diversity.py --config configs/cpt.yaml \
        --datasets droid xperience
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import av
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from compute_language_diversity import effective_rank

from reki.data.robotics.cameras import select_top_cameras
from reki.data.robotics.lerobot_io import resolve_lerobot_dirs
from reki.data.robotics.registry import DATASETS, spec_names
from reki.policy.train_config import TrainConfig

if TYPE_CHECKING:
    from transformers import BaseImageProcessor, PreTrainedModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

VideoFile = tuple[Path, np.ndarray, np.ndarray]
"""(mp4, start and end timestamps of its kept episodes)."""


def leaf_videos(leaf: Path, cam_map: dict[str, str]) -> list[VideoFile]:
    """The primary-camera video files of a leaf that hold kept episodes."""
    info = json.loads((leaf / "meta" / "info.json").read_text())
    videos = {k for k, v in info["features"].items() if v["dtype"] == "video"}
    key = select_top_cameras(cam_map, 1, available=videos)[0]
    col = f"videos/{key}/"
    cols = ["episode_index", *(col + c for c in ("chunk_index", "file_index"))]
    cols += [col + "from_timestamp", col + "to_timestamp"]
    eps = pd.concat(
        pq.read_table(f, columns=cols).to_pandas()
        for f in sorted((leaf / "meta" / "episodes").rglob("*.parquet"))
    )
    dropped_path = leaf / "meta" / "dropped_episodes.json"
    dropped = set(json.loads(dropped_path.read_text())) if dropped_path.exists() else set()
    eps = eps[~eps["episode_index"].isin(dropped)]
    out = []
    for (chunk, file), g in eps.groupby([col + "chunk_index", col + "file_index"]):
        rel = info["video_path"].format(video_key=key, chunk_index=chunk, file_index=file)
        g = g.sort_values(col + "from_timestamp")
        out.append(
            (leaf / rel, g[col + "from_timestamp"].to_numpy(), g[col + "to_timestamp"].to_numpy())
        )
    return out


def target_shape(
    mp4: Path, target_tokens: int, min_edge: int = 224, patch: int = 16
) -> tuple[int, int]:
    """Aspect-preserving size of about ``target_tokens`` patches, never upsampled past it and
    with a shortest edge of at least ``min_edge`` (DINOv3's pretraining resolution)."""
    container = av.open(str(mp4))
    h, w = container.streams.video[0].height, container.streams.video[0].width
    container.close()
    scale = max(min(1.0, (target_tokens * patch * patch / (h * w)) ** 0.5), min_edge / min(h, w))
    return max(patch, round(h * scale / patch) * patch), max(
        patch, round(w * scale / patch) * patch
    )


def decode_kept(
    video: VideoFile, stride: int, hw: tuple[int, int], stop: threading.Event
) -> np.ndarray:
    """Every ``stride``-th frame of a file that lies in a kept episode, ``[N, H, W, 3]`` uint8."""
    mp4, starts, ends = video
    frames = []
    with av.open(str(mp4)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for idx, frame in enumerate(container.decode(stream)):
            if stop.is_set():
                break
            if idx % stride:
                continue
            t = float(frame.time) + 1e-4
            ep = np.searchsorted(starts, t, side="right") - 1
            if ep >= 0 and t < ends[ep]:
                frames.append(
                    frame.reformat(width=hw[1], height=hw[0], format="rgb24").to_ndarray()
                )
    return np.stack(frames) if frames else np.zeros((0, *hw, 3), np.uint8)


def dataset_embeddings(
    name: str, model: PreTrainedModel, processor: BaseImageProcessor, args: argparse.Namespace
) -> np.ndarray:
    """DINOv3 CLS embeddings ``[N, D]`` of the sampled frames of one dataset."""
    files = [
        v for leaf in resolve_lerobot_dirs(name) for v in leaf_videos(leaf, DATASETS[name].cameras)
    ]
    random.Random(0).shuffle(files)
    hw = target_shape(files[0][0], args.target_tokens)
    logger.info("%s: %d files, frames resized to %dx%d", name, len(files), *hw)
    stop = threading.Event()
    out: list[np.ndarray] = []
    n = 0
    with ThreadPoolExecutor(args.workers) as ex:
        todo = iter(files)
        running: deque[Future[np.ndarray]] = deque(
            ex.submit(decode_kept, v, args.stride, hw, stop)
            for _, v in zip(range(args.workers), todo)
        )
        while running and n < args.max_frames:
            frames = running.popleft().result()[: args.max_frames - n]
            running.extend(
                ex.submit(decode_kept, v, args.stride, hw, stop) for v in [next(todo, None)] if v
            )
            for i in range(0, len(frames), args.batch):
                images = torch.from_numpy(frames[i : i + args.batch]).permute(0, 3, 1, 2)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    inputs = processor(
                        images=images, return_tensors="pt", device="cuda", do_resize=False
                    )
                    out.append(model(**inputs).last_hidden_state[:, 0].float().cpu().numpy())
            n += len(frames)
        stop.set()
    if not n:
        raise ValueError(f"{name}: no kept frames")
    return np.concatenate(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--datasets", nargs="*", help="default: the datasets of the config")
    parser.add_argument("--model", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--max-frames", type=int, default=50_000)
    parser.add_argument("--target-tokens", type=int, default=1200)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=32, help="files decoded in parallel")
    args = parser.parse_args()
    cfg = TrainConfig.from_config_and_overrides(args.config, {})
    if not cfg.vision_diversity_path:
        raise ValueError(f"{args.config}: no vision_diversity_path")
    names = args.datasets or spec_names(cfg.dataset)

    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).cuda().eval()
    out = Path(cfg.vision_diversity_path)
    for name in names:
        emb = dataset_embeddings(name, model, processor, args)
        score = effective_rank(emb, np.ones(len(emb)))
        logger.info("%s: %.3f over %d frames", name, score, len(emb))
        merged = {**json.loads(out.read_text()), name: score} if out.exists() else {name: score}
        merged = {k: round(v, 3) for k, v in sorted(merged.items(), key=lambda kv: kv[1])}
        out.write_text(json.dumps(merged, indent=2) + "\n")
    logger.info("wrote %d scores to %s", len(names), out)


if __name__ == "__main__":
    main()
