"""Rank helpers for distributed training."""

import os
from typing import Any


def is_rank_zero() -> bool:
    """True on global rank 0, from the launcher's ``RANK`` / ``LOCAL_RANK`` (True when unset)."""
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0


def print_main(*args: Any, **kwargs: Any) -> None:
    """``print`` on the main process only."""
    if is_rank_zero():
        print(*args, **kwargs)
