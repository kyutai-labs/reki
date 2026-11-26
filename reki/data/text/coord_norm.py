"""Rescale [0,1] coordinate strings to Qwen3.5's native [0,1000] [x,y] grounding
convention (RoboPoint/RefSpatial ship [0,1])."""

from __future__ import annotations

import re

_TUPLE_RE = re.compile(r"\(\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\s*\)")
_RANGE_HINT_RE = re.compile(r"between 0 and 1\b")


def _rescale(m: re.Match) -> str:
    x, y = float(m.group(1)), float(m.group(2))
    if x > 1.0 + 1e-6 or y > 1.0 + 1e-6:
        return m.group(0)
    return f"({round(x * 1000)}, {round(y * 1000)})"


def coords_unit_to_1000(text: str) -> str:
    """Rescale [0,1] coordinate tuples → [0,1000] integers; fix the range hint to match."""
    return _RANGE_HINT_RE.sub("between 0 and 1000", _TUPLE_RE.sub(_rescale, text))
