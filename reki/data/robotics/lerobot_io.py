"""Find and load the LeRobot (v3.0) directories of the registered datasets.

Every dataset lives under ``$REKI_DATASETS_ROOT`` (``reki.paths.datasets_root``): a single
repo under its spec's ``root``, or a collection of per-task repos under its spec's
``collection``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from reki.data.robotics.registry import DATASETS
from reki.paths import datasets_root
from reki.utils.dist_utils import is_rank_zero


def dataset_root(name: str) -> Path:
    """The directory that holds a single-repo dataset (the datasets root for an unknown name)."""
    return datasets_root() / DATASETS[name].root if name in DATASETS else datasets_root()


def collection_root(name: str) -> Path:
    """The root directory of a registered multi-repo collection."""
    collection = DATASETS[name].collection
    if collection is None:
        raise KeyError(f"{name!r} is not a multi-repo collection")
    return datasets_root() / collection


def is_collection(name: str) -> bool:
    """Whether ``name`` identifies a registered multi-repo collection."""
    return name in DATASETS and DATASETS[name].collection is not None


def find_collection_leaves(name: str) -> list[Path]:
    """The per-task LeRobot repos (``<task>/meta/info.json``) directly under a collection.
    Source directories suffixed ``_old`` by the v2.1-to-v3.0 converter are skipped."""
    root = collection_root(name)
    leaves = [
        info.parent.parent
        for info in sorted(root.glob("*/meta/info.json"))
        if not info.parent.parent.name.endswith("_old")
    ]
    if not leaves:
        raise FileNotFoundError(f"No LeRobot leaves for collection '{name}' under {root}")
    return leaves


def find_lerobot_dir(name: str, root: Path | None = None) -> Path:
    """The LeRobot directory of a single-repo dataset under ``root`` (default:
    ``dataset_root(name)``).

    Candidates are ``<name>_*`` directories with ``_lerobot`` in their name and
    ``<root>/<name>`` itself. Converter source directories suffixed ``_old`` are excluded.
    When several exist, the one with the highest ``codebase_version`` wins.
    """
    if root is None:
        root = dataset_root(name)
    candidates: list[Path] = []
    for candidate in root.glob(f"{name}_*"):
        if not candidate.is_dir():
            continue
        if "_lerobot" not in candidate.name or candidate.name.endswith("_old"):
            continue
        candidates.append(candidate)
    exact = root / name
    if exact.is_dir() and (exact / "meta" / "info.json").exists():
        candidates.append(exact)
    candidates = list(dict.fromkeys(candidates))
    if candidates:
        return max(candidates, key=_codebase_version)
    raise FileNotFoundError(f"No LeRobot dataset found for '{name}' in {root}")


def _codebase_version(path: Path) -> tuple[int, ...]:
    """The codebase version as a comparable tuple, or (0,) if unreadable."""
    try:
        info = json.load(open(path / "meta" / "info.json"))
        ver_str = info.get("codebase_version", "v0.0").lstrip("v")
        return tuple(int(x) for x in ver_str.split("."))
    except Exception:
        return (0,)


def resolve_lerobot_dirs(name: str) -> list[Path]:
    """All LeRobot repo dirs of a dataset: the leaves of a collection, else its one dir."""
    if is_collection(name):
        return find_collection_leaves(name)
    return [find_lerobot_dir(name, root=dataset_root(name))]


def training_leaves(dataset_names: list[str]) -> list[tuple[str, Path]]:
    """(dataset name, LeRobot leaf dir) of every leaf training loads, in load order. A
    collection gives one entry per per-task leaf."""
    return [(name, path) for name in dataset_names for path in resolve_lerobot_dirs(name)]


def load_lerobot_dir(name: str, lerobot_dir: Path, **kwargs: Any) -> Any:
    """Load one LeRobot repo dir and strip unused cameras."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from reki.data.robotics.lerobot_patches import install_lerobot_patches

    install_lerobot_patches()
    hf_log = logging.getLogger("huggingface_hub._snapshot_download")
    prev_level = hf_log.level
    hf_log.setLevel(logging.ERROR)
    try:
        ds = LeRobotDataset(repo_id=f"local/{name}", root=lerobot_dir, revision="local", **kwargs)
    finally:
        hf_log.setLevel(prev_level)

    if name in DATASETS:
        mapped = DATASETS[name].cameras
        unmapped = [k for k in ds.meta.camera_keys if k not in mapped]
        if len(unmapped) == len(ds.meta.camera_keys):
            raise ValueError(
                f"{name} ({lerobot_dir}): no camera of the dataset is in DATASETS[{name!r}].cameras "
                f"({ds.meta.camera_keys}). Map them to camera_top / camera_wrist_left / "
                "camera_wrist_right / camera_side_N (docs/data.md, Onboarding a dataset)."
            )
        for k in [k for k in ds.meta.video_keys if k not in mapped]:
            del ds.meta.features[k]
        if unmapped and is_rank_zero():
            print(f"{name}: cameras not in DATASETS[{name!r}].cameras are not used: {unmapped}")

    return ds


def leaf_meta_stats(name: str, lerobot_dir: Path) -> dict:
    """The raw ``meta.stats`` that ``load_lerobot_dir`` gives this leaf, without loading the
    data."""
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    return LeRobotDatasetMetadata(repo_id=f"local/{name}", root=lerobot_dir, revision="local").stats
