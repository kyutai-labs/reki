from typing import Any, Sequence

from reki.data.text.cached_arrow_dataset import CachedArrowDataset
from reki.data.text.llava_conversations import format_llava_conversations
from reki.paths import llava_ext_path


class LLava15OV(CachedArrowDataset):
    img_key = "image"
    conv_key = "conversations"

    def __init__(self, dataset_name: str, max_num_qa_pairs: int | None = None) -> None:
        super().__init__(llava_ext_path(), dataset_name, max_num_qa_pairs)

    def prepare_for_formatter(self, item: dict[str, Sequence[dict[str, str]]]) -> Any:
        return format_llava_conversations(item["conversations"], self.max_num_qa_pairs)
