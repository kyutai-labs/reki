"""Dataset registry: one ``DatasetSpec`` per training dataset, and the dataset groups.

A dataset spec in a config (``TrainConfig.dataset``) is a group name of ``DATASET_GROUPS``, a
dataset name of ``DATASETS``, or a comma-joined list of these (``expand_spec``). Adding a
dataset is one ``DATASETS`` entry (see docs/data.md, "Onboarding a dataset").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reki.policy.train_config import TrainConfig

logger = logging.getLogger(__name__)

DELTA_EE = "delta end-effector"
ABSOLUTE_JOINT = "absolute joint"
AXIS_ANGLE = "axis_angle"

MIRROR_SUFFIX = "_mirrored"


@dataclass(frozen=True)
class DatasetSpec:
    """How training reads one dataset.

    Slices pick (and order) the model-visible dims of the raw state/action vector; a ``None``
    entry in a state slice is a zero pad dim. Gripper flips index the sliced vector; their
    normalized value is negated, so that high means open for every dataset.
    """

    family: str
    """Source of the dataset; the sampling-mix printout sums over it."""
    control_mode: str
    cameras: dict[str, str]
    """Raw video key → canonical camera key. Several raw keys may map to one camera (the leaves
    of one collection name it differently); the first one a leaf has wins."""
    embodiment: str | None = None
    """State-MLP key (``TrainConfig.state_mlp_embodiments``); None trains without state."""
    root: Path = Path()
    """Directory under the datasets root that holds a single-repo dataset."""
    collection: Path | None = None
    """Multi-repo collection under the datasets root: one LeRobot repo per task under it."""
    state_slice: list[int | None] | None = None
    action_slice: list[int] | None = None
    robot_type: str | None = None
    """Prompt robot type. None: ``meta/info.json``'s robot type through ``ROBOT_TYPE_CANON``."""
    gripper_flip_state: list[int] | None = None
    gripper_flip_action: list[int] | None = None
    action_rot_rep: str | None = None
    """``AXIS_ANGLE``: the rotation is converted to euler xyz (``unify_ee_euler_rotation``)."""
    state_rot_rep: str | None = None
    recenter_state_roll: bool = False
    """Shift the euler roll of the state by π: a downward gripper has its roll near ±π, where it
    wraps (``recenters_state_roll``)."""
    mirror: str | None = None
    """Key of ``mirror.MIRROR_SPECS``: the dataset can be mirrored (``mirror_prob``)."""


def _cams(prefix: str = "observation.images.", **raw_to_camera: str) -> dict[str, str]:
    """``_cams(image="top")`` → ``{"observation.images.image": "observation.images.camera_top"}``."""
    return {prefix + raw: f"observation.images.camera_{cam}" for raw, cam in raw_to_camera.items()}


_ROBOTWIN_CAMS = _cams(cam_high="top", cam_left_wrist="wrist_left", cam_right_wrist="wrist_right")
_EE_EULER_STATE: list[int | None] = [0, 1, 2, 3, 4, 5, None, 6]

DATASETS: dict[str, DatasetSpec] = {
    "libero": DatasetSpec(
        "LIBERO",
        DELTA_EE,
        _cams(image="top", image2="wrist_left"),
        "single_arm_ee_euler",
        root=Path("HuggingFaceVLA"),
        state_slice=_EE_EULER_STATE,
        gripper_flip_action=[6],
        action_rot_rep=AXIS_ANGLE,
        state_rot_rep=AXIS_ANGLE,
        recenter_state_roll=True,
        mirror="libero_franka_y",
    ),
    "robotwin2_clean": DatasetSpec(
        "RoboTwin",
        ABSOLUTE_JOINT,
        _ROBOTWIN_CAMS,
        "aloha",
        collection=Path("robotwin2_clean_lerobot"),
        mirror="aloha_arx5",
    ),
    "robotwin2_randomized": DatasetSpec(
        "RoboTwin",
        ABSOLUTE_JOINT,
        _ROBOTWIN_CAMS,
        "aloha",
        collection=Path("robotwin2_randomized_lerobot") / "Randomized",
        mirror="aloha_arx5",
    ),
    "robocasa365": DatasetSpec(
        "RoboCasa",
        DELTA_EE,
        _cams(
            robot0_agentview_left="top",
            robot0_eye_in_hand="wrist_left",
            robot0_agentview_right="side_1",
        ),
        "panda_omron",
        collection=Path("robocasa365_lerobot_v30_collections") / "robocasa365",
        state_slice=list(range(0, 16)),
        action_slice=list(range(0, 12)),
        mirror="panda_omron_base_y",
    ),
}


def family(name: str) -> list[str]:
    """The datasets of one family, in registry order."""
    return [n for n, spec in DATASETS.items() if spec.family == name]


DATASET_GROUPS: dict[str, list[str]] = {
    "robotwin": family("RoboTwin"),
}
"""Groups of datasets that a config can name. A dataset name is a group of one dataset."""


def expand_spec(spec: str | list[str] | tuple[str, ...]) -> list[str]:
    """The dataset names of a dataset spec, de-duplicated in order. Raises on an unknown name;
    leaves out, with a warning, the datasets whose directory is missing; a group with none on
    disk raises."""
    if isinstance(spec, (list, tuple)) or "," in spec:
        parts = spec if isinstance(spec, (list, tuple)) else spec.split(",")
        out: list[str] = []
        for part in parts:
            for name in expand_spec(part):
                if name not in out:
                    out.append(name)
        return out
    if spec in DATASET_GROUPS:
        names = list(DATASET_GROUPS[spec])
    elif spec in DATASETS:
        names = [spec]
    else:
        raise ValueError(
            f"Unknown dataset group: {spec!r}. Available: {sorted(DATASET_GROUPS)} and the "
            "dataset names. Combine several with commas (e.g. 'robotwin2_clean,robotwin2_randomized')."
        )
    from reki.data.robotics.lerobot_io import resolve_lerobot_dirs

    found: list[str] = []
    missing: list[str] = []
    for name in names:
        try:
            resolve_lerobot_dirs(name)
        except FileNotFoundError as e:
            missing.append(str(e))
            continue
        found.append(name)
    if not found:
        raise FileNotFoundError(
            f"No datasets in spec {spec!r} could be located on disk:\n"
            + "\n".join(missing)
            + "\nSee docs/data.md#benchmark-datasets."
        )
    if missing:
        logger.warning(
            "Skipping %d dataset(s) of %r with no on-disk directory:\n%s",
            len(missing),
            spec,
            "\n".join(missing),
        )
    return found


def state_embodiments(names: list[str]) -> set[str]:
    """State-MLP keys of the registered datasets among ``names``."""
    return {e for n in names if n in DATASETS and (e := DATASETS[n].embodiment) is not None}


def mirrored_name(base: str) -> str:
    """Mirror-world counterpart of a state-embodiment key or a robot type."""
    return f"{base}{MIRROR_SUFFIX}"


def mirror_state_embodiments(dataset_names: list[str], mirror_prob: float) -> set[str]:
    """Extra state-MLP keys that the mirror augmentation adds for these datasets.

    A mirrored sample uses the key ``mirrored_name(<the dataset's key>)``. With
    ``0 < mirror_prob < 1`` one dataset uses two MLPs, and both must exist. Each mirrorable
    embodiment in the mix adds one key. Raises when ``mirror_prob > 0`` and nothing in the mix
    is mirrorable, or when a mirrorable dataset has no state embodiment.
    """
    if mirror_prob <= 0.0:
        return set()
    mirrorable = [n for n in dataset_names if n in DATASETS and DATASETS[n].mirror]
    if not mirrorable:
        raise ValueError(
            f"mirror_prob={mirror_prob} but none of {sorted(dataset_names)} is mirrorable "
            f"(supported: {mirrorable_datasets()})."
        )
    embodiments = [DATASETS[n].embodiment for n in mirrorable]
    missing = [n for n, e in zip(mirrorable, embodiments, strict=True) if e is None]
    if missing:
        raise ValueError(
            f"mirrorable datasets {missing} have no state embodiment; the mirrored sample "
            "would have no MLP to route to (the mirror needs a separate bank by design)"
        )
    return {mirrored_name(emb) for emb in embodiments if emb is not None}


def recenters_state_roll(name: str, cfg: TrainConfig) -> bool:
    """Whether the euler roll of ``name``'s state is shifted by π, in training and at eval
    (``transforms.recenter_euler_roll``, stats from the ``norm_stats_euler_roll.json`` sidecar)."""
    spec = DATASETS[name]
    return (
        cfg.unify_ee_euler_rotation
        and spec.state_rot_rep == AXIS_ANGLE
        and spec.recenter_state_roll
    )


def mirrorable_datasets() -> list[str]:
    """The registered mirrorable dataset names in sorted order."""
    return sorted(n for n, spec in DATASETS.items() if spec.mirror)


ROBOT_TYPE_CANON: dict[str, str] = {
    "panda": "franka",
    "GR1ArmsAndWaistFourierHands": "gr1",
}
"""``meta/info.json`` robot type → the robot type the prompt names."""


def canon_robot_type(robot_type: str | None, dataset_name: str | None = None) -> str | None:
    """The prompt robot type of a dataset: its spec's ``robot_type``, else the canonical name
    of its ``meta/info.json`` robot type."""
    if dataset_name in DATASETS and (override := DATASETS[dataset_name].robot_type):
        return override
    if robot_type is None:
        return None
    return ROBOT_TYPE_CANON.get(robot_type, robot_type)
