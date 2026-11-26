"""Convert FineVision box coordinates to plain-text ``[0,1000]`` integer xyxy, the Qwen3-VL
grounding convention (``coord_norm`` does the same for points).

Applied per QA turn in ``CachedArrowFineVision.prepare_for_formatter``, so the source parquet
is never rewritten. ``objects365_qa`` ships ``[0,1]`` center boxes ``(cx, cy, w, h)`` (every
sampled box is valid as a center box, only ~71% as top-left). The transform is idempotent: a
4-tuple with values outside ``[0,1]`` is already converted and is left as is.
"""

from __future__ import annotations

import re
from typing import Callable

_BBOX4_RE = re.compile(
    r"\(\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\s*\)"
)


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def _cxcywh_unit_to_xyxy_1000(m: re.Match) -> str:
    cx, cy, w, h = (float(m.group(i)) for i in range(1, 5))
    if max(cx, cy, w, h) > 1.0 + 1e-6:
        return m.group(0)
    x1 = round(_clamp01(cx - w / 2) * 1000)
    y1 = round(_clamp01(cy - h / 2) * 1000)
    x2 = round(_clamp01(cx + w / 2) * 1000)
    y2 = round(_clamp01(cy + h / 2) * 1000)
    return f"({x1}, {y1}, {x2}, {y2})"


def _reformat_objects365(text: str) -> str:
    text = text.replace(
        "normalized coordinates (x, y, w, h)",
        "coordinates (x1, y1, x2, y2), each between 0 and 1000",
    )
    text = text.replace("(x, y, w, h)", "(x1, y1, x2, y2)")
    return _BBOX4_RE.sub(_cxcywh_unit_to_xyxy_1000, text)


FINEVISION_COORD_REFORMAT: dict[str, Callable[[str], str]] = {
    "objects365_qa": _reformat_objects365,
}


def reformat_text(dataset_name: str, text: str) -> str:
    """Apply the config's coordinate transform to one QA-turn text (no-op if none)."""
    fn = FINEVISION_COORD_REFORMAT.get(dataset_name)
    return fn(text) if fn is not None else text
