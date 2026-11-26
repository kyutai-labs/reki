"""The reader interface and the per-dataset recipe.

A family module ``readers/<family>.py`` defines its readers and two dicts keyed by registry
dataset: ``PINS`` (``pins.Pin``) and ``RECIPES`` (``Recipe``). ``catalog`` collects them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .common import dedup, download
from .pins import Pin

if TYPE_CHECKING:
    from reki.policy.train_config import TrainConfig

StatsOp = Callable[[str, Path, list[Path], bool], None]
"""A stats step, called with the registry name, the datasets root, the dataset's leaves and
whether this is a ``--subset`` build."""
HorizonOp = Callable[[str, Path, list[Path], bool, "TrainConfig"], None]
"""A stats step that depends on the training horizon: a ``StatsOp`` that also takes the
training config (chunk length, action pace)."""


def root_dir(name: str, root: Path) -> Path:
    """``<root>/<registry root>`` of dataset ``name``."""
    from reki.data.robotics.registry import DATASETS

    return root / DATASETS[name].root


def collection_dir(name: str, root: Path) -> Path:
    """``<root>/<registry collection>`` of dataset ``name``."""
    from reki.data.robotics.registry import DATASETS

    collection = DATASETS[name].collection
    assert collection is not None
    return root / collection


class Reader:
    """Builds the leaves of one dataset from its downloaded upstream. ``pin`` is the full pin or,
    with ``--subset``, the subset pin."""

    finish_after_shards = False
    """A ``build`` shard writes no leaves; the unsharded ``build`` writes them. The last shard to
    end runs it."""

    def leaves(self, name: str, root: Path, pin: Pin) -> list[Path]:
        """The output leaves of dataset ``name`` under the datasets root ``root``."""
        raise NotImplementedError

    def download(self, name: str, root: Path, pin: Pin) -> None:
        """Fetch every source of ``pin`` into ``<root>/upstream/``."""
        for source in pin.sources:
            download.fetch(source, download.upstream_dir(root, source))

    def build(self, name: str, root: Path, pin: Pin, shard: tuple[int, int]) -> None:
        """Write the leaves (shard ``(i, n)`` of the work; ``(0, 1)``: all of it)."""
        raise NotImplementedError

    def curate(self, name: str, root: Path, pin: Pin) -> None:
        """Reader-specific curation outputs (e.g. subtask lookups), after the detectors."""

    def dedup_labels(self, name: str, root: Path, pin: Pin, leaf: Path) -> list[str] | None:
        """Per episode of ``leaf``, the label that names the kept copy of duplicates in the
        list of excluded episodes (``common/drops.py``); None: the task text."""
        return None

    def key_columns(self) -> dedup.Columns | None:
        """The ``(state, action)`` columns of the episode key (``common/dedup.py``); None: all.
        A reader whose values can differ across hardware or software keys on its stable
        columns, so that the list of excluded episodes matches on every machine."""
        return None


class VlReader(Reader):
    """Builds one vision-language dataset: QA rows with images, one dir per config under
    ``<root>/<out>`` (``reki.paths``), not LeRobot leaves. Only download, build and verify
    apply; ``shard`` splits the configs."""

    out: str

    def verify(self, name: str, root: Path, pin: Pin) -> None:
        """Check the built configs: row counts and the schema the loader reads."""
        raise NotImplementedError


@dataclass(frozen=True)
class Recipe:
    """How one registry dataset is built. ``curate`` names ``detect.DETECTORS``; ``stats`` is
    the ordered list of ``stats.op`` steps; ``horizon`` runs after them, with the training
    config."""

    reader: Reader
    curate: tuple[str, ...]
    stats: tuple[StatsOp, ...]
    horizon: tuple[HorizonOp, ...] = ()
