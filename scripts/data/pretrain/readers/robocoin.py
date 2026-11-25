"""RoboCOIN: 672 upstream LeRobot v2.1 task repos (``RoboCOIN/<task>``, one commit each) → one
v3.0 leaf each, at ``<root>/robocoin_lerobot_v30_collections/<collection>/<leaf>``.

``pins_robocoin.tsv`` holds, per leaf: its collection, the repo id and commit, the meta totals
and the duplicate counts (full pin, ``--subset`` pin). The repos are a snapshot of 2026-06-04:
repos created later, or empty then (``Galaxea_R1_Lite_pour_powder``), are not used.

Data transforms: ``AIRBOT_MMK2_*`` leaves (and ``Airbot_MMK2_storage_peach_pear``) store the hand
dims 12-35 of state and action in degrees; they are converted to radians before any stats.
``press_the_button_b`` has 65 episode files but its meta lists 34: its meta is rebuilt from the
data. Only the core columns are kept (the ``annotation.*``, ``eef_*``, ``gripper_*`` columns go).
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ..common import v30
from ..common.download import upstream_dir
from ..common.stats import op
from ..pins import HfSnapshot, Pin
from ..recipe import Reader, Recipe, collection_dir
from .lerobot_v2x import V2X_COLUMNS

logger = logging.getLogger(__name__)

PINS_TSV = Path(__file__).parents[1] / "pins_robocoin.tsv"
ALLOW = ("meta/**", "data/**", "videos/**")

HAND_DIMS = slice(12, 36)
DEGREE_THRESHOLD = 6.5
"""A hand dim above this is in degrees: radian leaves stay below 2.1, degree episodes reach 58+."""
DEG2RAD = np.float32(np.pi / 180)
REBUILD_META = ("alpha_bot_2_press_the_button_b",)


@dataclass(frozen=True)
class LeafPin:
    leaf: str
    collection: str
    repo: str
    revision: str
    episodes: int
    frames: int
    in_subset: bool
    """The leaf is one of the ``--subset`` leaves."""


def leaf_pins() -> list[LeafPin]:
    with open(PINS_TSV, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return [
        LeafPin(
            r["leaf"],
            r["collection"],
            r["repo"],
            r["revision"],
            int(r["episodes"]),
            int(r["frames"]),
            r["subset"] == "yes",
        )
        for r in rows
    ]


def collection_of(name: str, state_dim: int, grip_idx: tuple[int, ...]) -> str | None:
    """The collection of a leaf, from its name, state width and gripper dims (state layouts
    differ between collections). None: a leaf left out (19 R1_Lite leaves with state_dim != 14)."""
    if name.startswith("Galbot"):
        return "robocoin_galbot"
    if name.startswith(("Airbot", "AIRBOT_MMK2")):
        return "robocoin_airbot_mmk2"
    if name.startswith(("Agilex", "Cobot_Magic", "Split_aloha")):
        if state_dim == 14:
            return "robocoin_agilex_14"
        if grip_idx == (6, 19):
            return "robocoin_agilex_26"
        if grip_idx == (12, 25):
            return "robocoin_agilex_26b"
        raise ValueError(f"{name}: unrecognized 26-dim agilex gripper layout {grip_idx}")
    if name.startswith("Galaxea_R1_Lite"):
        return "robocoin_galaxea_r1lite"
    if name.startswith("R1_Lite"):
        return "robocoin_r1lite" if state_dim == 14 else None
    if name.startswith("G1edu"):
        return "robocoin_unitree_g1_28" if state_dim == 28 else "robocoin_unitree_g1_30"
    if name.startswith(("AgiBot-g1", "Tianqin")):
        return "robocoin_a2d_17" if state_dim == 17 else "robocoin_a2d_41"
    if name.startswith(("RMC-AIDA-L", "Realman")):
        return "robocoin_realman_schemaA"
    if name.startswith(("alpha_bot_2", "AI2_Alphabot_2")):
        return "robocoin_realman_schemaB"
    if name.startswith("leju_robot"):
        return "robocoin_leju"
    return None


def hand_degrees(src: Path) -> bool:
    """Whether an airbot leaf stores its hand dims in degrees. Every episode must agree."""
    per_episode = [
        max(np.abs(ep.state[:, HAND_DIMS]).max(), np.abs(ep.action[:, HAND_DIMS]).max())
        > DEGREE_THRESHOLD
        for ep in v30.read_v2x_episodes(src)
    ]
    if len(set(per_episode)) != 1:
        raise ValueError(f"{src}: {sum(per_episode)}/{len(per_episode)} episodes in degrees")
    return per_episode[0]


def to_radians(state: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    state, action = state.copy(), action.copy()
    state[:, HAND_DIMS] *= DEG2RAD
    action[:, HAND_DIMS] *= DEG2RAD
    return state, action


def _leaf_of(pin: Pin) -> dict[str, LeafPin]:
    """Repo id → leaf pin, for the sources of ``pin``."""
    by_repo = {p.repo: p for p in leaf_pins()}
    return {s.repo: by_repo[s.repo] for s in pin.sources if isinstance(s, HfSnapshot)}


class RoboCoin(Reader):
    def leaves(self, name: str, root: Path, pin: Pin) -> list[Path]:
        return [collection_dir(name, root) / p.leaf for p in _leaf_of(pin).values()]

    def build(self, name: str, root: Path, pin: Pin, shard: tuple[int, int]) -> None:
        i, n = shard
        jobs = list(zip(pin.sources, _leaf_of(pin).values(), self.leaves(name, root, pin)))
        for source, p, out in jobs[i::n]:
            src = upstream_dir(root, source)
            rebuild = p.leaf in REBUILD_META
            lengths = v30.pinned_lengths(src, p.episodes, p.frames, rebuild)
            info = json.loads((src / "meta" / "info.json").read_text())
            state = info["features"]["observation.state"]
            grip_idx = tuple(j for j, s in enumerate(state.get("names") or []) if "gripper" in s)
            routed = collection_of(p.leaf, state["shape"][0], grip_idx)
            if routed != name:
                raise ValueError(f"{p.leaf}: routes to {routed}, pinned in {name}")
            transform = to_radians if name == "robocoin_airbot_mmk2" and hand_degrees(src) else None
            v30.convert_v2x(src, out, list(lengths), V2X_COLUMNS, transform, rebuild)
            logger.info(
                "%s: %d episodes → %s%s",
                p.leaf,
                len(lengths),
                out,
                " (hand dims degrees → radians)" if transform else "",
            )


def _pin(leaves: list[LeafPin]) -> Pin:
    return Pin(
        tuple(HfSnapshot(p.repo, p.revision, ALLOW) for p in leaves),
        episodes=sum(p.episodes for p in leaves),
        frames=sum(p.frames for p in leaves),
    )


def _pins() -> dict[str, Pin]:
    by_collection: dict[str, list[LeafPin]] = {}
    for p in sorted(leaf_pins(), key=lambda p: p.leaf):
        by_collection.setdefault(p.collection, []).append(p)
    out = {}
    for name, leaves in by_collection.items():
        subset = [p for p in leaves if p.in_subset]
        out[name] = replace(_pin(leaves), subset=_pin(subset))
    return out


PINS = _pins()

_DEFAULT = (op("kept_qk"), op("global_qk"))
_STATS = {
    "robocoin_agilex_26": (
        op("kept_qk"),
        op("global_qk", pool=("robocoin_agilex_26", "robocoin_agilex_26b")),
    ),
    "robocoin_leju": (
        op("kept_qk"),
        op("global_qk", columns=("action",)),
        op("global_qk", columns=("observation.state",), pool_dims=40),
    ),
    "robocoin_realman_schemaA": (op("kept_qk"), op("global_qk", clip_min=0.0, clip_dims=(7, 21))),
    "robocoin_realman_schemaB": (
        op("kept_qk"),
        op("global_qk", pool_dims=28, clip_min=0.0, clip_dims=(26, 27)),
    ),
}
"""agilex_26: band pooled with agilex_26b, as trained (agilex_26b keeps its own band). leju: state
widths differ (118/54), dims 0-39 align. Realman: encoder dims clipped at 0;
schemaB widths differ (28/34), dims 0-27 align."""

RECIPES = {
    name: Recipe(RoboCoin(), curate=("frozen", "short_video"), stats=_STATS.get(name, _DEFAULT))
    for name in PINS
}
