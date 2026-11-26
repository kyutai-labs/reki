"""GR00T-X-Embodiment-Sim (``nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim``): 11 upstream
LeRobot v2.0 subsets → one v3.0 leaf each, at ``<root>/<spec.collection>/<subset>``.

No data transforms. Only the core columns are kept: the ``annotation.*`` / ``next.*`` columns
are dropped (their declared feature types do not match storage; training reads none of them);
``frame_index``, absent upstream, is added. Panda euler stats are one band over the collection
(the trained sidecars).
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

from ..common import v30
from ..common.download import upstream_dir
from ..common.stats import op
from ..pins import HfSnapshot, Pin
from ..recipe import Reader, Recipe, collection_dir
from .lerobot_v2x import V2X_COLUMNS

logger = logging.getLogger(__name__)

REPO = "nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"
REVISION = "ea7ac0b68f87da62f1e726771bba0fe74300802f"
SUBSETS: dict[str, dict[str, tuple[int, int]]] = {
    "groot_gr1": {
        "gr1_arms_only.CanSort": (1000, 316878),
        "gr1_arms_waist.CanToDrawer": (10107, 3210426),
        "gr1_arms_waist.CupToDrawer": (10036, 2664870),
        "gr1_arms_waist.CuttingboardToBasket": (10056, 2411674),
        "gr1_arms_waist.CuttingboardToCardboardBox": (10042, 2002060),
    },
    "groot_panda_gripper": {
        "bimanual_panda_gripper.Threading": (1000, 213493),
        "bimanual_panda_gripper.ThreePieceAssembly": (1000, 238372),
        "bimanual_panda_gripper.Transport": (1000, 409386),
    },
    "groot_panda_hand": {
        "bimanual_panda_hand.BoxCleanup": (1000, 230689),
        "bimanual_panda_hand.DrawerCleanup": (1000, 290716),
        "bimanual_panda_hand.LiftTray": (1000, 500131),
    },
}
"""Per dataset: upstream subset → (episodes, frames) of its pinned ``meta/info.json``."""

SUBSET = {
    "groot_gr1": "gr1_arms_only.CanSort",
    "groot_panda_gripper": "bimanual_panda_gripper.Threading",
    "groot_panda_hand": "bimanual_panda_hand.BoxCleanup",
}
"""``--subset``: the smallest subset of each collection."""


def _pin(subsets: list[str], counts: dict[str, tuple[int, int]]) -> Pin:
    return Pin(
        (HfSnapshot(REPO, REVISION, tuple(f"{s}/**" for s in subsets)),),
        episodes=sum(counts[s][0] for s in subsets),
        frames=sum(counts[s][1] for s in subsets),
    )


def subsets_of(pin: Pin) -> list[str]:
    """The upstream subsets a pin downloads."""
    (source,) = pin.sources
    assert isinstance(source, HfSnapshot) and source.allow is not None
    return [pattern.removesuffix("/**") for pattern in source.allow]


class Groot(Reader):
    def leaves(self, name: str, root: Path, pin: Pin) -> list[Path]:
        return [collection_dir(name, root) / s for s in subsets_of(pin)]

    def build(self, name: str, root: Path, pin: Pin, shard: tuple[int, int]) -> None:
        up = upstream_dir(root, pin.sources[0])
        i, n = shard
        jobs = list(zip(subsets_of(pin), self.leaves(name, root, pin)))
        for sub, out in jobs[i::n]:
            lengths = v30.pinned_lengths(up / sub, *SUBSETS[name][sub])
            v30.convert_v2x(up / sub, out, list(lengths), V2X_COLUMNS)
            logger.info("%s: %d episodes → %s", name, len(lengths), out)


PINS = {
    name: replace(_pin(list(counts), counts), subset=_pin([SUBSET[name]], counts))
    for name, counts in SUBSETS.items()
}

RECIPES = {
    "groot_gr1": Recipe(Groot(), curate=("short_video",), stats=(op("kept_qk"), op("global_qk"))),
    "groot_panda_gripper": Recipe(
        Groot(), curate=("short_video",), stats=(op("kept_qk"), op("euler", pooled=True))
    ),
    "groot_panda_hand": Recipe(
        Groot(), curate=("short_video",), stats=(op("kept_qk"), op("euler", pooled=True))
    ),
}
