"""VL spatial datasets loaded from the FineVision-format Arrow shards (or parquet) at
``reki.paths.vl_spatial_parquet_path()`` (built by ``scripts/data/pretrain_data.py``; embedded
images, ``[0,1000]`` coords, PixMo QA pre-synthesized). ``images`` is a list → ``VLSampleAdapter`` inserts every image
into the first user turn (multi-view safe).
"""

from __future__ import annotations

from typing import Sequence

from reki.data.text.cached_arrow_dataset import CachedArrowDataset
from reki.data.text.llava_conversations import format_llava_conversations
from reki.paths import vl_spatial_parquet_path


class VLSpatialParquet(CachedArrowDataset):
    conv_key = "conversations"

    def __init__(self, dataset_name: str, max_num_qa_pairs: int | None = None) -> None:
        super().__init__(vl_spatial_parquet_path(), dataset_name, max_num_qa_pairs)

    def prepare_for_formatter(
        self, item: dict[str, Sequence[dict[str, str]]]
    ) -> list[tuple[int, str]]:
        return format_llava_conversations(item["conversations"], self.max_num_qa_pairs)
