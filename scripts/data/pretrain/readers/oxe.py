"""Open X-Embodiment: eleven IPEC-COMMUNITY LeRobot v2.x repos (``lerobot_v2x``) and bridge_v2
(``Gaugou/BridgeV2``, already v3.0, ``lerobot_v2x.PassthroughV30``). berkeley_rpt_delta is in
``franka_delta.py``.

No data transforms. Leaf names are the on-disk ones, so the loader resolves the same folders.
hydra, nyu_franka, sailor, mutex and viola keep their ``_0.1.0_lerobot`` names.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..common.stats import op
from ..pins import HfSnapshot, Pin
from ..recipe import Recipe
from .lerobot_v2x import V2X_COLUMNS, LeRobotV2x, PassthroughV30, episode_files


@dataclass(frozen=True)
class _V2x:
    repo: str
    revision: str
    leaf: str
    video_keys: tuple[str, ...]
    episodes: int
    frames: int
    subset: tuple[int, ...]
    """Upstream episodes of ``--subset``: episode 0, or one same-text and one different-text
    duplicate group."""
    subset_frames: int
    frozen: bool

    def pin(self) -> Pin:
        sub = Pin(
            (HfSnapshot(self.repo, self.revision, episode_files(self.subset, self.video_keys)),),
            episodes=len(self.subset),
            frames=self.subset_frames,
            select=self.subset,
        )
        return Pin(
            (HfSnapshot(self.repo, self.revision),),
            episodes=self.episodes,
            frames=self.frames,
            subset=sub,
        )


_IMAGE_WRIST = ("observation.images.image", "observation.images.wrist_image")

V2X: dict[str, _V2x] = {
    "taco_play": _V2x(
        "IPEC-COMMUNITY/taco_play_lerobot",
        "f46f2478738d33cbc67317d879aa091567f0db8e",
        "taco_play_hf_lerobot",
        ("observation.images.rgb_static", "observation.images.rgb_gripper"),
        3242, 213972, (0,), 66, frozen=False,
    ),
    "stanford_hydra_dataset_converted_externally_to_rlds": _V2x(
        "IPEC-COMMUNITY/stanford_hydra_dataset_lerobot",
        "beaa981ca3d2ee7860b1acb837f29911c7eef725",
        "stanford_hydra_dataset_converted_externally_to_rlds_0.1.0_lerobot",
        _IMAGE_WRIST,
        570, 358234, (0,), 549, frozen=False,
    ),
    "nyu_franka_play_dataset_converted_externally_to_rlds": _V2x(
        "IPEC-COMMUNITY/nyu_franka_play_dataset_lerobot",
        "1e3b48c92d367dbca044c8fb5f1106d1f1be0542",
        "nyu_franka_play_dataset_converted_externally_to_rlds_0.1.0_lerobot",
        ("observation.images.image", "observation.images.image_additional_view"),
        365, 34448, (0,), 42, frozen=False,
    ),
    "bc_z": _V2x(
        "IPEC-COMMUNITY/bc_z_lerobot",
        "ddc4da6f3baa0196e42d22a177d038274b0ba8ae",
        "bc_z_hf_lerobot_v30",
        ("observation.images.image",),
        39350, 5471693, (44, 36180), 402, frozen=True,
    ),
    "fractal20220817_data": _V2x(
        "IPEC-COMMUNITY/fractal20220817_data_lerobot",
        "91bf7d7f7ce50770a1ba5c6db14b8d1c0815122e",
        "fractal20220817_data_hf_lerobot_v30",
        ("observation.images.image",),
        87212, 3786400, (1, 53760, 62750, 74444), 136, frozen=True,
    ),
    "furniture_bench_dataset_converted_externally_to_rlds": _V2x(
        "IPEC-COMMUNITY/furniture_bench_dataset_lerobot",
        "00ce24057a680b7e8f10aab3e6203edf21499384",
        "furniture_bench_hf_lerobot_v30",
        _IMAGE_WRIST,
        5100, 3948057, (0,), 438, frozen=False,
    ),
    "austin_sailor_dataset_converted_externally_to_rlds": _V2x(
        "IPEC-COMMUNITY/austin_sailor_dataset_lerobot",
        "da8ad6a93d288df66d022387eeb13bc6e55d1d61",
        "austin_sailor_dataset_converted_externally_to_rlds_0.1.0_lerobot",
        _IMAGE_WRIST,
        240, 353094, (0,), 2377, frozen=False,
    ),
    "austin_buds_dataset_converted_externally_to_rlds": _V2x(
        "IPEC-COMMUNITY/austin_buds_dataset_lerobot",
        "466d7a73e992208693b825a9ed31efa4c06da9e3",
        "austin_buds_dataset_hf_lerobot",
        _IMAGE_WRIST,
        50, 34112, (0,), 640, frozen=False,
    ),
    "utaustin_mutex": _V2x(
        "IPEC-COMMUNITY/utaustin_mutex_lerobot",
        "cced76715b3322c69b4096e3a6a16de1d862207f",
        "utaustin_mutex_0.1.0_lerobot",
        _IMAGE_WRIST,
        1500, 361883, (0,), 169, frozen=False,
    ),
    "berkeley_cable_routing": _V2x(
        "IPEC-COMMUNITY/berkeley_cable_routing_lerobot",
        "2ce23db5a4da14ce88d5678b8d384f5c376625b8",
        "berkeley_cable_routing_hf_lerobot",
        (
            "observation.images.top_image",
            "observation.images.image",
            "observation.images.wrist45_image",
            "observation.images.wrist225_image",
        ),
        1482, 38240, (0,), 25, frozen=True,
    ),
    "viola": _V2x(
        "IPEC-COMMUNITY/viola_lerobot",
        "4808352b4ff3fb8985a91fa7ee12b4d8a80717f3",
        "viola_0.1.0_lerobot",
        ("observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"),
        135, 68913, (0,), 481, frozen=False,
    ),
}  # fmt: skip
"""Per dataset: upstream at its pinned commit, output leaf, video keys,
the pinned meta totals, the duplicates ``dedup`` removes, and the ``--subset`` slice."""


BRIDGE_REPO = "Gaugou/BridgeV2"
BRIDGE_REVISION = "ad12b2a476e0107e1b245de76c2a33a52b1181a2"
"""BridgeData V2 ``demos_8_17`` (rail-berkeley, CC BY 4.0)."""
BRIDGE_SUBSET = (12577, 12604, 12610, 12889, 12966, 24541, 24542)
"""A different-text pair (bowl on plate / ... cardboard fence), a same-text pair (turn faucet
left) and three episodes that ``curate`` keeps, for a dry run: data file 0 and 8 video files."""
BRIDGE_SUBSET_FILES = (
    "meta/**",
    "data/chunk-000/file-000.parquet",
    "videos/observation.images.primary/chunk-000/file-039.mp4",
    "videos/observation.images.primary/chunk-000/file-064.mp4",
    "videos/observation.images.side1/chunk-000/file-009.mp4",
    "videos/observation.images.side1/chunk-000/file-032.mp4",
    "videos/observation.images.side2/chunk-000/file-000.mp4",
    "videos/observation.images.side2/chunk-000/file-004.mp4",
    "videos/observation.images.wrist/chunk-000/file-009.mp4",
    "videos/observation.images.wrist/chunk-000/file-032.mp4",
)

PINS: dict[str, Pin] = {name: spec.pin() for name, spec in V2X.items()} | {
    "bridge_v2": Pin(
        (HfSnapshot(BRIDGE_REPO, BRIDGE_REVISION),),
        episodes=50418,
        frames=1801162,
        subset=Pin(
            (HfSnapshot(BRIDGE_REPO, BRIDGE_REVISION, BRIDGE_SUBSET_FILES),),
            episodes=7,
            frames=212,
            select=BRIDGE_SUBSET,
        ),
    ),
}

RECIPES: dict[str, Recipe] = {
    name: Recipe(
        LeRobotV2x(spec.leaf, V2X_COLUMNS),
        curate=("frozen", "short_video") if spec.frozen else ("short_video",),
        stats=(op("kept_qk"),),
    )
    for name, spec in V2X.items()
} | {
    "bridge_v2": Recipe(
        PassthroughV30("bridge_v2_lerobot"),
        curate=("frozen", "short_video"),
        stats=(op("kept_qk"),),
    ),
}
