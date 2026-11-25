"""Dataset registry: one ``DatasetSpec`` per training dataset, and the dataset groups.

A dataset spec in a config (``TrainConfig.dataset``) is a group name of ``DATASET_GROUPS``, a
dataset name of ``DATASETS``, or a comma-joined list of these (``expand_spec``). Adding a
dataset is one ``DATASETS`` entry (see docs/data.md, "Onboarding a dataset").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

if TYPE_CHECKING:
    from reki.policy.train_config import TrainConfig

DELTA_EE = "delta end-effector"
ABSOLUTE_JOINT = "absolute joint"
RELATIVE_EE = "relative end-effector"
AXIS_ANGLE = "axis_angle"
SINGLE_ARM: Final = "single_arm"
"""``arm_action_dims`` of one arm with two wrist cameras: never judged static."""

OXE_DIR = Path("open_x_embodiment_lerobot")
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
    dir: str | None = None
    """LeRobot folder of a single-repo dataset under ``root``; None finds it by name
    (``lerobot_io.find_lerobot_dir``)."""
    collection: Path | None = None
    """Multi-repo collection under the datasets root: one LeRobot repo per task under it."""
    state_slice: list[int | None] | None = None
    action_slice: list[int] | None = None
    robot_type: str = field(kw_only=True)
    """Robot type in the prompt; ``meta/info.json``'s robot type is not read."""
    gripper_flip_state: list[int] | None = None
    gripper_flip_action: list[int] | None = None
    action_rot_rep: str | None = None
    """``AXIS_ANGLE``: the rotation is converted to euler xyz (``unify_ee_euler_rotation``)."""
    state_rot_rep: str | None = None
    recenter_state_roll: bool = False
    """Shift the euler roll of the state by π: a downward gripper has its roll near ±π, where it
    wraps (``recenters_state_roll``)."""
    gripper_unit_index: int | None = None
    """Raw gripper index whose quantile band is re-stamped to its physical unit
    (``normalization.apply_gripper_unit``)."""
    weight_group: str | None = None
    """Datasets of one weight group count as one source in the power-law weighting."""
    mirror: str | None = None
    """Key of ``mirror.MIRROR_SPECS``: the dataset can be mirrored (``mirror_prob``)."""
    arm_action_dims: tuple[list[int], list[int]] | Literal["single_arm"] | None = None
    """(left arm, right arm) dims of the sliced action, gripper/hand included; the static-wrist
    camera drop (``cameras.wrist_static_flags``) reads them. Set iff both wrist cameras are
    mapped: ``SINGLE_ARM`` for one arm with two wrist cameras."""
    subtask_lookup: bool = False
    """The dataset has a subtask lookup (``TrainConfig.subtask_lookup_filename``) in its leaves.
    With ``predict_subtask``, a mismatch with the files on disk stops the run at the start."""


def _cams(prefix: str = "observation.images.", **raw_to_camera: str) -> dict[str, str]:
    """``_cams(image="top")`` → ``{"observation.images.image": "observation.images.camera_top"}``."""
    return {prefix + raw: f"observation.images.camera_{cam}" for raw, cam in raw_to_camera.items()}


_IMAGE_AND_WRIST = _cams(image="top", wrist_image="wrist_left")
_ONLY_TOP = _cams(camera_top="top")
_EGO_VIEW = _cams(ego_view="top")
_HIGH_AND_WRISTS = _cams(
    cam_high_rgb="top", cam_left_wrist_rgb="wrist_left", cam_right_wrist_rgb="wrist_right"
)
_ROBOTWIN_CAMS = _cams(cam_high="top", cam_left_wrist="wrist_left", cam_right_wrist="wrist_right")
_INTERNDATA_BIMANUAL_CAMS = _cams(
    "images.rgb.", head="top", hand_left="wrist_left", hand_right="wrist_right"
)
_ABC_CAMS = _cams(
    camera_top="top", camera_wrist_left="wrist_left", camera_wrist_right="wrist_right"
)
_AGILEX_26_CAMS = _cams(
    cam_head_rgb="top",
    cam_high_rgb="top",
    cam_front_chest_rgb="side_1",
    cam_high_realsense_rgb="side_2",
    cam_left_wrist_rgb="wrist_left",
    cam_right_wrist_rgb="wrist_right",
)

_EE_EULER_STATE: list[int | None] = [0, 1, 2, 3, 4, 5, None, 6]
_GR1 = [*range(0, 13), *range(22, 35), *range(41, 44)]
_AIRBOT_MMK2 = [*range(0, 14), *range(17, 24, 2), 24, 25, *range(29, 36, 2)]
_AGILEX_26 = [0, 1, 2, 3, 4, 5, 6, 13, 14, 15, 16, 17, 18, 19]
_AGILEX_26B = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 25]
_A2D_41 = [0, 1, 2, 3, 4, 5, 6, 32, 7, 8, 9, 10, 11, 12, 13, 33]
_A2D_17 = [1, 2, 3, 4, 5, 6, 7, 15, 8, 9, 10, 11, 12, 13, 14, 16]
_REALMAN_A = [14, 15, 16, 17, 18, 19, 20, 21, 0, 1, 2, 3, 4, 5, 6, 7]
_REALMAN_B = [0, 1, 2, 3, 4, 5, 6, 26, 13, 14, 15, 16, 17, 18, 19, 27]
_UNITREE_G1_30 = list(range(28))
_LEJU = [*range(0, 14), *range(26, 38)]

_ARMS_7_7 = (list(range(0, 7)), list(range(7, 14)))
_ARMS_8_8 = (list(range(0, 8)), list(range(8, 16)))
_ARMS_DEX_26 = ([*range(0, 7), *range(14, 20)], [*range(7, 14), *range(20, 26)])
_ARMS_R1LITE_GROUPED = ([*range(0, 6), 12], [*range(6, 12), 13])
_ARMS_UNITREE_G1 = ([*range(0, 7), *range(14, 21)], [*range(7, 14), *range(21, 28)])

_ROBOMIND = Path("robomind_lerobot")
_ROBOCOIN = Path("robocoin_lerobot_v30_collections")
_GROOT = Path("groot_x_embodiment_sim_lerobot_v30_collections")
_INTERNDATA = Path("interndata_a1_lerobot_v30_collections")
_EGO = Path("egocentric_raw")

_oxe = partial(DatasetSpec, "OXE", DELTA_EE, root=OXE_DIR)
_robomind = partial(DatasetSpec, "RoboMIND", ABSOLUTE_JOINT, root=_ROBOMIND)
_robocoin = partial(DatasetSpec, "RoboCOIN", ABSOLUTE_JOINT)
_ego = partial(DatasetSpec, "Ego", RELATIVE_EE, _ONLY_TOP, robot_type="human_hand")

DATASETS: dict[str, DatasetSpec] = {
    # Open X-Embodiment. docs/data.md lists the excluded sets and why.
    "taco_play": _oxe(
        _cams(rgb_static="top", rgb_gripper="wrist_left"),
        "single_arm_ee_euler",
        robot_type="franka",
    ),
    "stanford_hydra_dataset_converted_externally_to_rlds": _oxe(
        _IMAGE_AND_WRIST,
        "single_arm_ee_euler",
        robot_type="franka",
    ),
    "nyu_franka_play_dataset_converted_externally_to_rlds": _oxe(
        _cams(image="top", image_additional_view="side_1"),
        "single_arm_ee_euler",
        robot_type="franka",
    ),
    "bc_z": _oxe(
        _cams(image="top"),
        "single_arm_ee_euler",
        gripper_flip_state=[7],
        robot_type="google_robot",
    ),
    "fractal20220817_data": _oxe(
        _cams(image="top"),
        "single_arm_ee_quat",
        gripper_flip_state=[7],
        robot_type="google_robot",
    ),
    "furniture_bench_dataset_converted_externally_to_rlds": _oxe(
        _IMAGE_AND_WRIST,
        "single_arm_ee_quat",
        dir="furniture_bench_hf_lerobot_v30",
        robot_type="franka",
    ),
    "austin_sailor_dataset_converted_externally_to_rlds": _oxe(
        _IMAGE_AND_WRIST,
        "single_arm_ee_quat",
        robot_type="franka",
    ),
    "austin_buds_dataset_converted_externally_to_rlds": _oxe(
        _IMAGE_AND_WRIST,
        "franka",
        dir="austin_buds_dataset_hf_lerobot",
        robot_type="franka",
    ),
    "utaustin_mutex": _oxe(
        _IMAGE_AND_WRIST,
        "franka",
        robot_type="franka",
    ),
    "berkeley_cable_routing": _oxe(
        _cams(
            top_image="top",
            image="side_1",
            wrist45_image="wrist_left",
            wrist225_image="wrist_right",
        ),
        "franka",
        robot_type="franka",
        arm_action_dims=SINGLE_ARM,
    ),
    "viola": _oxe(
        _cams(agentview_rgb="top", eye_in_hand_rgb="wrist_left"),
        "franka",
        robot_type="franka",
    ),
    "bridge_v2": _oxe(
        _cams(primary="top", wrist="wrist_left", side1="side_1", side2="side_2"),
        "single_arm_ee_euler",
        state_slice=_EE_EULER_STATE,
        robot_type="widowx",
    ),
    "berkeley_rpt_delta": _oxe(
        _cams(hand_image="wrist_left"),
        "single_arm_ee_euler",
        gripper_flip_state=[7],
        gripper_flip_action=[6],
        action_rot_rep=AXIS_ANGLE,
        state_rot_rep=AXIS_ANGLE,
        robot_type="franka",
    ),
    "droid": DatasetSpec(
        "DROID",
        ABSOLUTE_JOINT,
        _cams(exterior_1_left="top", exterior_2_left="side_1", wrist_left="wrist_left"),
        "franka",
        root=OXE_DIR,
        robot_type="franka",
        gripper_flip_state=[7],
        gripper_flip_action=[7],
    ),
    "agibot_world_2026": DatasetSpec(
        "AgiBot",
        ABSOLUTE_JOINT,
        _cams(
            top_head="top",
            hand_left="wrist_left",
            hand_right="wrist_right",
            head_left_fisheye="side_1",
            head_right_fisheye="side_2",
        ),
        "g2a",
        robot_type="agibot_g2",
        arm_action_dims=([0, *range(2, 9)], [1, *range(9, 16)]),
        state_slice=[0, 1, *range(40, 54), *range(82, 85), *range(85, 90)],
        action_slice=[0, 1, *range(16, 30), *range(33, 38)],
        subtask_lookup=True,
    ),
    "robomind_franka_1rgb": _robomind(
        _ONLY_TOP,
        "franka",
        state_slice=list(range(8)),
        gripper_flip_state=[7],
        gripper_flip_action=[7],
        weight_group="robomind_franka",
        robot_type="franka",
    ),
    "robomind_franka_3rgb": _robomind(
        _cams(camera_top="top", camera_left="side_1", camera_right="side_2"),
        "franka",
        state_slice=list(range(8)),
        gripper_flip_state=[7],
        gripper_flip_action=[7],
        weight_group="robomind_franka",
        robot_type="franka",
    ),
    "robomind_ur_1rgb": _robomind(
        _ONLY_TOP,
        "ur5e",
        state_slice=list(range(7)),
        gripper_flip_state=[6],
        gripper_flip_action=[6],
        robot_type="ur5e",
    ),
    "robomind_franka_fr3_dual": _robomind(
        _cams(camera_front="top", camera_top="side_1", camera_left="side_2", camera_right="side_3"),
        "bimanual_joint",
        state_slice=list(range(16)),
        gripper_flip_state=[7, 15],
        gripper_flip_action=[7, 15],
        robot_type="franka_fr3",
    ),
    "robomind_agilex_3rgb": _robomind(
        _cams(camera_front="top", camera_left_wrist="wrist_left", camera_right_wrist="wrist_right"),
        "agilex",
        gripper_flip_state=[6, 13],
        gripper_flip_action=[6, 13],
        robot_type="agilex",
        arm_action_dims=_ARMS_7_7,
    ),
    "robomind_tienkung_gello_1rgb": _robomind(
        _ONLY_TOP,
        "bimanual_joint",
        robot_type="tienkung",
        weight_group="robomind_tienkung_gello",
    ),
    "robomind_tienkung_prod1_gello_1rgb": _robomind(
        _ONLY_TOP,
        "bimanual_joint",
        robot_type="tienkung",
        weight_group="robomind_tienkung_gello",
    ),
    "robomind_tienkung_xsens_1rgb": _robomind(
        _ONLY_TOP, "bimanual_dexterous", robot_type="tienkung_dex"
    ),
    "galaxea": DatasetSpec(
        "Galaxea",
        ABSOLUTE_JOINT,
        _cams(
            head_rgb="top",
            head_right_rgb="side_1",
            left_wrist_rgb="wrist_left",
            right_wrist_rgb="wrist_right",
        ),
        "r1lite",
        state_slice=[*range(0, 6), *range(12, 18), 24, 25, 26, 48, 49],
        action_slice=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 21, 25],
        gripper_flip_state=[15, 16],
        gripper_flip_action=[12, 13],
        robot_type="r1lite",
        arm_action_dims=_ARMS_R1LITE_GROUPED,
        subtask_lookup=True,
    ),
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
        robot_type="franka",
    ),
    "robotwin2_clean": DatasetSpec(
        "RoboTwin",
        ABSOLUTE_JOINT,
        _ROBOTWIN_CAMS,
        "aloha",
        collection=Path("robotwin2_clean_lerobot"),
        mirror="aloha_arx5",
        robot_type="aloha",
        arm_action_dims=_ARMS_7_7,
    ),
    "robotwin2_randomized": DatasetSpec(
        "RoboTwin",
        ABSOLUTE_JOINT,
        _ROBOTWIN_CAMS,
        "aloha",
        collection=Path("robotwin2_randomized_lerobot") / "Randomized",
        mirror="aloha_arx5",
        robot_type="aloha",
        arm_action_dims=_ARMS_7_7,
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
        robot_type="PandaOmron",
    ),
    "roboset_delta": DatasetSpec(
        "RoboSet",
        DELTA_EE,
        _cams(top="top", wrist="wrist_left", left="side_1", right="side_2"),
        "single_arm_ee_euler",
        collection=Path("roboset_delta_lerobot"),
        action_rot_rep=AXIS_ANGLE,
        state_rot_rep=AXIS_ANGLE,
        robot_type="franka",
    ),
    "rdt": DatasetSpec(
        "RDT",
        ABSOLUTE_JOINT,
        _cams(high="top", left_wrist="wrist_left", right_wrist="wrist_right"),
        "aloha",
        collection=Path("rdt_lerobot"),
        gripper_flip_state=[6, 13],
        gripper_flip_action=[6, 13],
        robot_type="aloha",
        arm_action_dims=_ARMS_7_7,
    ),
    "abc_yam_realsense": DatasetSpec(
        "ABC",
        ABSOLUTE_JOINT,
        _ABC_CAMS,
        "yam",
        collection=Path("abc_yam_realsense_lerobot"),
        weight_group="abc_yam",
        robot_type="yam",
        arm_action_dims=_ARMS_7_7,
        subtask_lookup=True,
    ),
    "abc_yam_zedx": DatasetSpec(
        "ABC",
        ABSOLUTE_JOINT,
        _ABC_CAMS,
        "yam",
        collection=Path("abc_yam_zedx_lerobot"),
        weight_group="abc_yam",
        robot_type="yam",
        arm_action_dims=_ARMS_7_7,
    ),
    "robocoin_galbot": _robocoin(
        _HIGH_AND_WRISTS,
        "bimanual_joint",
        collection=_ROBOCOIN / "robocoin_galbot",
        state_slice=list(range(5, 21)),
        robot_type="galbot",
        arm_action_dims=_ARMS_8_8,
    ),
    "robocoin_airbot_mmk2": _robocoin(
        _cams(
            cam_head_rgb="top",
            cam_high_rgb="top",
            cam_front_rgb="side_1",
            cam_third_view="side_1",
            cam_left_wrist_rgb="wrist_left",
            cam_right_wrist_rgb="wrist_right",
        ),
        "airbot_mmk2",
        collection=_ROBOCOIN / "robocoin_airbot_mmk2",
        state_slice=list(_AIRBOT_MMK2),
        action_slice=_AIRBOT_MMK2,
        robot_type="airbot_mmk2",
        arm_action_dims=([*range(0, 6), *range(12, 18)], [*range(6, 12), *range(18, 24)]),
    ),
    "robocoin_agilex_14": _robocoin(
        _cams(
            cam_front_rgb="top",
            cam_high_rgb="top",
            cam_left_wrist_rgb="wrist_left",
            cam_left_wrist_rgb_rgb="wrist_left",
            cam_right_wrist_rgb="wrist_right",
            cam_right_wrist_rgb_rgb="wrist_right",
        ),
        "agilex",
        collection=_ROBOCOIN / "robocoin_agilex_14",
        robot_type="agilex",
        arm_action_dims=_ARMS_7_7,
        gripper_flip_state=[6, 13],
        gripper_flip_action=[6, 13],
        weight_group="robocoin_agilex",
    ),
    "robocoin_agilex_26": _robocoin(
        _AGILEX_26_CAMS,
        "agilex",
        collection=_ROBOCOIN / "robocoin_agilex_26",
        state_slice=list(_AGILEX_26),
        action_slice=_AGILEX_26,
        robot_type="agilex",
        arm_action_dims=_ARMS_7_7,
        weight_group="robocoin_agilex",
    ),
    "robocoin_agilex_26b": _robocoin(
        _AGILEX_26_CAMS,
        "agilex",
        collection=_ROBOCOIN / "robocoin_agilex_26b",
        state_slice=list(_AGILEX_26B),
        action_slice=_AGILEX_26B,
        robot_type="agilex",
        arm_action_dims=_ARMS_7_7,
        weight_group="robocoin_agilex",
    ),
    "robocoin_r1lite": _robocoin(
        _HIGH_AND_WRISTS,
        "r1lite",
        collection=_ROBOCOIN / "robocoin_r1lite",
        state_slice=[*range(0, 6), *range(7, 13), *([None] * 3), 6, 13],
        robot_type="r1lite",
        arm_action_dims=_ARMS_7_7,
        gripper_flip_state=[15, 16],
        gripper_flip_action=[6, 13],
        weight_group="robocoin_r1lite",
    ),
    "robocoin_galaxea_r1lite": _robocoin(
        _cams(
            cam_head_left_rgb="top",
            cam_head_right_rgb="side_1",
            cam_left_wrist_rgb="wrist_left",
            cam_right_wrist_rgb="wrist_right",
        ),
        "r1lite",
        collection=_ROBOCOIN / "robocoin_galaxea_r1lite",
        state_slice=[*range(0, 12), *([None] * 3), 12, 13],
        robot_type="r1lite",
        arm_action_dims=_ARMS_R1LITE_GROUPED,
        weight_group="robocoin_r1lite",
    ),
    "robocoin_a2d_41": _robocoin(
        _HIGH_AND_WRISTS,
        "bimanual_joint",
        collection=_ROBOCOIN / "robocoin_a2d_41",
        state_slice=list(_A2D_41),
        action_slice=_A2D_41,
        robot_type="agibot_a2",
        arm_action_dims=_ARMS_8_8,
        weight_group="robocoin_a2d",
    ),
    "robocoin_a2d_17": _robocoin(
        _HIGH_AND_WRISTS,
        "bimanual_joint",
        collection=_ROBOCOIN / "robocoin_a2d_17",
        state_slice=list(_A2D_17),
        action_slice=_A2D_17,
        robot_type="agibot_a2",
        arm_action_dims=_ARMS_8_8,
        weight_group="robocoin_a2d",
    ),
    "robocoin_realman_schemaA": _robocoin(
        _cams(
            cam_high_rgb="top",
            cam_head_rgb="top",
            cam_left_wrist_rgb="wrist_left",
            cam_right_wrist_rgb="wrist_right",
        ),
        "bimanual_joint",
        collection=_ROBOCOIN / "robocoin_realman_schemaA",
        state_slice=list(_REALMAN_A),
        action_slice=_REALMAN_A,
        robot_type="realman",
        arm_action_dims=_ARMS_8_8,
        weight_group="robocoin_realman_aidal",
    ),
    "robocoin_realman_schemaB": _robocoin(
        _cams(
            cam_head_rgb="top",
            cam_front_head_rgb="top",
            cam_chest_rgb="side_1",
            cam_front_chest_rgb="side_1",
            cam_left_wrist_rgb="wrist_left",
            cam_right_wrist_rgb="wrist_right",
        ),
        "bimanual_joint",
        collection=_ROBOCOIN / "robocoin_realman_schemaB",
        state_slice=list(_REALMAN_B),
        action_slice=_REALMAN_B,
        robot_type="alphabot",
        arm_action_dims=_ARMS_8_8,
        weight_group="robocoin_realman_aidal",
    ),
    "robocoin_unitree_g1_28": _robocoin(
        _cams(
            cam_high_rgb="top",
            cam_left_high="top",
            color_left_wrist="wrist_left",
            cam_left_wrist="wrist_left",
            color_right_wrist="wrist_right",
            cam_right_wrist="wrist_right",
        ),
        "unitree_g1",
        collection=_ROBOCOIN / "robocoin_unitree_g1_28",
        robot_type="unitree_g1",
        arm_action_dims=_ARMS_UNITREE_G1,
        weight_group="robocoin_unitree_g1",
    ),
    "robocoin_unitree_g1_30": _robocoin(
        _EGO_VIEW,
        "unitree_g1",
        collection=_ROBOCOIN / "robocoin_unitree_g1_30",
        state_slice=list(_UNITREE_G1_30),
        action_slice=_UNITREE_G1_30,
        robot_type="unitree_g1",
        weight_group="robocoin_unitree_g1",
    ),
    "robocoin_leju": _robocoin(
        _cams(
            camera_head_rgb="top",
            camera_left_wrist_rgb="wrist_left",
            camera_right_wrist_rgb="wrist_right",
        ),
        "bimanual_dexterous",
        collection=_ROBOCOIN / "robocoin_leju",
        state_slice=list(_LEJU),
        action_slice=_LEJU,
        robot_type="kuavo",
        arm_action_dims=_ARMS_DEX_26,
    ),
    "egodex": _ego(collection=_EGO / "egodex_lerobot_v30"),
    "egoverse": _ego(collection=_EGO / "egoverse_lerobot_v30"),
    "xperience": _ego(collection=_EGO / "xperience_lerobot_v30", subtask_lookup=True),
    "groot_gr1": DatasetSpec(
        "GR00T",
        ABSOLUTE_JOINT,
        _EGO_VIEW,
        "gr1",
        collection=_GROOT / "groot_gr1",
        robot_type="gr1",
        state_slice=list(_GR1),
        action_slice=_GR1,
    ),
    "groot_panda_gripper": DatasetSpec(
        "GR00T",
        DELTA_EE,
        _cams(front_view="top", left_wrist_view="wrist_left", right_wrist_view="wrist_right"),
        "bimanual_joint",
        collection=_GROOT / "groot_panda_gripper",
        robot_type="bimanual_panda",
        arm_action_dims=_ARMS_7_7,
        state_slice=[*range(7, 14), 14, *range(23, 30), 30],
        gripper_flip_state=[7, 15],
        gripper_flip_action=[6, 13],
        action_rot_rep=AXIS_ANGLE,
    ),
    "groot_panda_hand": DatasetSpec(
        "GR00T",
        DELTA_EE,
        _cams(ego_view="top", left_wrist_view="wrist_left", right_wrist_view="wrist_right"),
        "bimanual_dexterous",
        collection=_GROOT / "groot_panda_hand",
        robot_type="bimanual_panda_hand",
        arm_action_dims=(list(range(0, 11)), list(range(11, 22))),
        state_slice=[*range(7, 14), *range(27, 34), *range(14, 20), *range(34, 40)],
        action_slice=[*range(0, 11), *range(12, 23)],
        action_rot_rep=AXIS_ANGLE,
    ),
    "interndata_franka": DatasetSpec(
        "InternData",
        DELTA_EE,
        _cams("images.rgb.", head="top", hand="wrist_left"),
        "single_arm_ee_euler",
        collection=_INTERNDATA / "interndata_franka",
        robot_type="franka",
        state_slice=_EE_EULER_STATE,
        action_rot_rep=AXIS_ANGLE,
        gripper_unit_index=6,
    ),
    "interndata_split_aloha": DatasetSpec(
        "InternData",
        ABSOLUTE_JOINT,
        _INTERNDATA_BIMANUAL_CAMS,
        "aloha",
        collection=_INTERNDATA / "interndata_split_aloha",
        robot_type="agilex_piper",
        arm_action_dims=_ARMS_7_7,
    ),
    "interndata_genie1": DatasetSpec(
        "InternData",
        ABSOLUTE_JOINT,
        _INTERNDATA_BIMANUAL_CAMS,
        "bimanual_joint",
        collection=_INTERNDATA / "interndata_genie1",
        robot_type="genie1",
        arm_action_dims=_ARMS_8_8,
    ),
}


def family(name: str) -> list[str]:
    """The datasets of one family, in registry order."""
    return [n for n, spec in DATASETS.items() if spec.family == name]


DATASET_GROUPS: dict[str, list[str]] = {
    "oxe_reduced": family("OXE"),
    "robomind": family("RoboMIND"),
    "robotwin": family("RoboTwin"),
    "abc": family("ABC"),
    "robocoin": family("RoboCOIN"),
    "egocentric": family("Ego"),
    "groot": family("GR00T"),
    "interndata": family("InternData"),
}
"""Groups of datasets that a config can name. A dataset name is a group of one dataset."""

VAL_DATASETS: dict[str, list[int] | None] = {"libero": None}
"""Datasets usable as ``val_dataset`` → the episodes held out of training, as written by
``scripts/data/make_val_split.py``; None validates on every episode."""


def spec_names(spec: str | list[str] | tuple[str, ...]) -> list[str]:
    """The dataset names of a dataset spec, de-duplicated in order, without a disk check.
    Raises on an unknown name."""
    if isinstance(spec, (list, tuple)) or "," in spec:
        parts = spec if isinstance(spec, (list, tuple)) else spec.split(",")
        out: list[str] = []
        for part in parts:
            for name in spec_names(part):
                if name not in out:
                    out.append(name)
        return out
    if spec in DATASET_GROUPS:
        return list(DATASET_GROUPS[spec])
    if spec in DATASETS:
        return [spec]
    raise ValueError(
        f"Unknown dataset group: {spec!r}. Available: {sorted(DATASET_GROUPS)} and the "
        "dataset names. Combine several with commas (e.g. 'oxe_reduced,droid')."
    )


def expand_spec(spec: str | list[str] | tuple[str, ...]) -> list[str]:
    """``spec_names`` of a dataset spec. Raises when any dataset of it is missing on disk."""
    from reki.data.robotics.lerobot_io import resolve_lerobot_dirs

    names = spec_names(spec)
    missing: list[str] = []
    for name in names:
        try:
            resolve_lerobot_dirs(name)
        except FileNotFoundError as e:
            missing.append(str(e))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} dataset(s) of spec {spec!r} could not be located on disk:\n"
            + "\n".join(missing)
            + "\nFor a subset of a group, list the wanted datasets by name"
            " (e.g. 'robotwin2_clean,libero').\nSee docs/data.md#benchmark-datasets."
        )
    return names


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
