"""REKI: a VLA policy on Qwen3.5. ``from reki import RekiPolicy`` is the inference API.

The import is lazy, so ``import reki`` does not load torch and the model code.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from reki.inference.policy import RekiPolicy

__all__ = ["RekiPolicy"]


def __getattr__(name: str) -> Any:
    if name == "RekiPolicy":
        from reki.inference.policy import RekiPolicy

        return RekiPolicy
    raise AttributeError(f"module 'reki' has no attribute {name!r}")
