"""Every dataset's pin and recipe, collected from the ``PINS`` / ``RECIPES`` dicts of each
``readers/`` module, and the groups of datasets outside the registry
(``GROUPS``). A name defined twice raises."""

from __future__ import annotations

import importlib
import pkgutil
from functools import cache
from types import ModuleType

from . import readers
from .pins import Pin
from .recipe import Recipe, VlReader


def _modules() -> list[ModuleType]:
    return [
        importlib.import_module(f"{readers.__name__}.{m.name}")
        for m in pkgutil.iter_modules(readers.__path__)
    ]


def _merge(tables: list[dict]) -> dict:
    """Union of the tables; a table imported into several modules counts once."""
    out: dict = {}
    for table in {id(t): t for t in tables}.values():
        twice = set(out) & set(table)
        if twice:
            raise KeyError(f"datasets defined twice: {sorted(twice)}")
        out |= table
    return out


@cache
def pins() -> dict[str, Pin]:
    """Pin per registry dataset."""
    return _merge([getattr(m, "PINS", {}) for m in _modules()])


@cache
def recipes() -> dict[str, Recipe]:
    """Recipe per registry dataset."""
    return _merge([getattr(m, "RECIPES", {}) for m in _modules()])


def pin(name: str, subset: bool) -> Pin:
    """The pin a stage uses: the subset pin with ``subset`` when the dataset has one."""
    full = pins()[name]
    return full.subset if subset and full.subset is not None else full


@cache
def groups() -> dict[str, list[str]]:
    """Groups of datasets outside the registry (e.g. ``vl``)."""
    return _merge([getattr(m, "GROUPS", {}) for m in _modules()])


def is_vl(name: str) -> bool:
    """Whether ``name`` is a vision-language dataset (not LeRobot leaves)."""
    recipe = recipes().get(name)
    return recipe is not None and isinstance(recipe.reader, VlReader)
