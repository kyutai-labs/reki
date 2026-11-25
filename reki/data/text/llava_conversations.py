"""Shared LLaVA-style ``conversations`` → ``[(speaker_id, text), ...]`` formatter.

Both ``LLava15OV`` and ``VLSpatialParquet`` consume the identical LLaVA schema
(``{"from": "human"|"gpt", "value": ...}`` or ``{"role": "user"|"assistant", "content":
...}``).
"""

from __future__ import annotations

import random
import re
from typing import Sequence

import numpy as np
import pyarrow as pa


def alternating_rows(conversations: pa.ChunkedArray) -> np.ndarray:
    """Mask of the rows whose ``conversations`` is non-empty and alternates user, assistant,
    user, ... with the per-row key choice of ``format_llava_conversations`` (rows of one config
    can mix the ``from`` and ``role`` schemas). Reads only the speaker fields."""
    return np.concatenate(
        [_alternating_chunk(chunk) for chunk in conversations.chunks] or [np.zeros(0, dtype=bool)]
    )


def _alternating_chunk(conv: pa.ListArray) -> np.ndarray:
    turns = conv.flatten()
    keep = conv.value_lengths().fill_null(0).to_numpy() > 0
    parents = conv.value_parent_indices().to_numpy()
    first = conv.offsets.to_numpy()[:-1] - conv.offsets[0].as_py()
    position = np.arange(len(turns)) - first[parents]

    def field(name: str) -> np.ndarray:
        if name not in turns.type.names:
            return np.full(len(turns), None, dtype=object)
        return turns.field(name).to_numpy(zero_copy_only=False)

    speaker_from, speaker_role = field("from"), field("role")
    row_uses_from = np.zeros(len(conv), dtype=bool)
    row_uses_from[keep] = speaker_from[first[keep]] != None  # noqa: E711
    uses_from = row_uses_from[parents]
    speaker = np.where(uses_from, speaker_from, speaker_role)
    even = position % 2 == 0
    expected = np.where(
        uses_from, np.where(even, "human", "gpt"), np.where(even, "user", "assistant")
    )
    keep[parents[speaker != expected]] = False
    return keep


def format_llava_conversations(
    texts: Sequence[dict[str, str]],
    max_num_qa_pairs: int | None,
) -> list[tuple[int, str]]:
    """Normalize a LLaVA-style ``conversations`` list into ``[(speaker_id, text)]``.

    ``speaker_id`` is ``1`` for the user and ``0`` for the assistant. Handles both
    the ``{from, value}`` (human/gpt) and ``{role, content}`` (user/assistant)
    schemas, asserts strict user→assistant alternation, optionally subsamples
    ``max_num_qa_pairs`` consecutive Q&A pairs, strips ``<image>`` placeholders,
    and removes a leading ``"Revised answer:"`` prefix from assistant turns.
    """
    value_key, role_key, user_key, assistant_key = (
        ("value", "from", "human", "gpt")
        if "from" in texts[0] and texts[0]["from"] is not None
        else ("content", "role", "user", "assistant")
    )
    for i in range(len(texts)):
        if i % 2 == 0:
            assert texts[i][role_key] == user_key, f"Expected {user_key}, got {texts[i][role_key]}"
        else:
            assert texts[i][role_key] == assistant_key, (
                f"Expected {assistant_key}, got {texts[i][role_key]}"
            )
    if max_num_qa_pairs is not None and len(texts) // 2 > max_num_qa_pairs:
        idcs = random.sample(range(len(texts) // 2), max_num_qa_pairs)
        sampled_texts = []
        for idx in sorted(idcs):
            sampled_texts.append(texts[2 * idx])
            sampled_texts.append(texts[2 * idx + 1])
        texts = sampled_texts

    def _remove_revised_answer_part(s: str) -> str:
        pattern = r"^(\s+)?Revised answer:(\s+)?"
        return re.sub(pattern, "", s, flags=re.IGNORECASE)

    return [
        (
            (idx + 1) % 2,
            re.sub(r"<image>", "", _remove_revised_answer_part(x[value_key])),
        )
        for idx, x in enumerate(texts)
    ]
