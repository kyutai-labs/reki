"""Base class for vision-language QA datasets read from one config dir on disk."""

import logging
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from torch.utils.data import Dataset

from reki.data.text.llava_conversations import alternating_rows
from reki.data.utils import load_vl_config

logger = logging.getLogger(__name__)


class CachedArrowDataset(Dataset, ABC):
    """Base for HF question-answering datasets (train split, decoded images). Rows with a null
    image are dropped at load (``svit-part-09-of-10`` ships 22k of them), and, with ``conv_key``,
    rows whose LLaVA conversation does not alternate user and assistant (one in ``llrv_gpt4v``)."""

    img_key = "images"
    conv_key: str | None = None

    def __init__(self, root: str, dataset_name: str, max_num_qa_pairs: int | None = None):
        self.dataset_name = dataset_name
        ds = load_vl_config(root, dataset_name)
        images = ds.data.column(self.img_key)
        keep = images.is_valid().to_numpy(zero_copy_only=False)
        if images.null_count:
            logger.info("%s: dropping %d rows without image", dataset_name, images.null_count)
        if self.conv_key is not None:
            alternating = alternating_rows(ds.data.column(self.conv_key))
            if not alternating.all():
                logger.info(
                    "%s: dropping %d rows whose conversation does not alternate user and assistant",
                    dataset_name,
                    (~alternating).sum(),
                )
            keep &= alternating
        if not keep.all():
            ds = ds.select(np.flatnonzero(keep))
        self.ds = ds
        self.max_num_qa_pairs = max_num_qa_pairs

    def __repr__(self):
        return repr(self.ds)

    def __len__(self) -> int:
        return len(self.ds)

    @abstractmethod
    def prepare_for_formatter(
        self, item: dict[str, Sequence[dict[str, str]]]
    ) -> Sequence[tuple[int, str]]:
        """Turn one HF sample into a conversation: a list of ``(speaker_id, text)``,
        with ``1`` for the user and ``0`` for the assistant."""
        raise NotImplementedError("Abstract method")
