"""FineVision datasets. FineVision has only a train split."""

import random
from typing import Sequence

from reki.data.text.cached_arrow_dataset import CachedArrowDataset
from reki.paths import finevision_path


class CachedArrowFineVision(CachedArrowDataset):
    def __init__(self, dataset_name: str, max_num_qa_pairs: int | None = None):
        super().__init__(finevision_path(), dataset_name, max_num_qa_pairs)

    def prepare_for_formatter(
        self, item: dict[str, Sequence[dict[str, str]]]
    ) -> Sequence[tuple[int, str]]:
        texts = item["texts"]
        if self.max_num_qa_pairs is not None and len(texts) > self.max_num_qa_pairs:
            texts = random.sample(texts, self.max_num_qa_pairs)
        from reki.data.text.coord_reformat import reformat_text

        name = self.dataset_name
        return [
            x
            for cap in texts
            for x in [
                (1, reformat_text(name, cap["user"])),
                (0, reformat_text(name, cap["assistant"])),
            ]
        ]
