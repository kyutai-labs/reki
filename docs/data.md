# Data

All robot data uses the LeRobot v3.0 format. Each dataset is one `DatasetSpec` in
[`reki/data/robotics/registry.py`](../reki/data/robotics/registry.py), with the dataset groups.
The mixture weights are in [`reki/data/robotics/sampling.py`](../reki/data/robotics/sampling.py)
([pretraining mixture](pretraining_data.md#pretraining-mixture)).
[Benchmark datasets](#benchmark-datasets) gives the steps to build the benchmark datasets.

## State formats

Each dataset has a state format (`DatasetSpec.embodiment`): the key of its state MLP in
`state_mlp_embodiments`. Datasets that share a key must share the state dim and the meaning and
order of each dim. The key follows the format, not the brand (AgiBot G2A and A2D differ), and
formats with the same dims stay separate when the kinematics differ (`agilex`, `aloha`, `yam`).
A dataset without a key is trained without state.

Actions do not have to match the state format. The actions of each dataset are normalized and
supervised separately, padded with zeros to 32 dims.

| Key | Dim | Layout | Example datasets (pretraining or benchmark) |
|---|---|---|---|
| `single_arm_ee_euler` | 8 | xyz, rpy, pad, gripper | `bridge_v2`, `bc_z`, `libero`, `interndata_franka` |
| `single_arm_ee_quat` | 8 | xyz, quaternion (xyzw), gripper | `fractal20220817_data`, `furniture_bench_dataset_converted_externally_to_rlds` |
| `franka` | 8 | 7 joints, gripper | `droid`, `robomind_franka_1rgb` |
| `ur5e` | 7 | 6 joints, gripper | `robomind_ur_1rgb` |
| `g2a` | 24 | AgiBot G2A end effectors, joints, head, waist | `agibot_world_2026` |
| `bimanual_joint` | 16 | [left arm 7, left gripper, right arm 7, right gripper] | `robomind_franka_fr3_dual`, `interndata_genie1` |
| `bimanual_dexterous` | 26 | [left arm 7, right arm 7, left hand 6, right hand 6] | `groot_panda_hand`, `robocoin_leju` |
| `agilex` | 14 | [left arm 6, left gripper, right arm 6, right gripper] | `robomind_agilex_3rgb` |
| `aloha` | 14 | [left arm 6, left gripper, right arm 6, right gripper] | `robotwin2_clean`, `rdt`, `interndata_split_aloha` |
| `yam` | 14 | [left arm 6, left gripper, right arm 6, right gripper] | `abc_yam_realsense`, `abc_yam_zedx` |
| `r1lite` | 17 | arms 2×6, torso 3, grippers 2 | `galaxea`, `robocoin_r1lite` |
| `gr1` | 29 | [left arm 7, left hand 6, right arm 7, right hand 6, waist 3] | `groot_gr1` |
| `airbot_mmk2` | 24 | arm and hand joints of both arms | `robocoin_airbot_mmk2` |
| `unitree_g1` | 28 | arm and hand joints of both arms | `robocoin_unitree_g1_28`, `robocoin_unitree_g1_30` |
| `panda_omron` | 16 | base pose (3+4), end-effector pose in the base frame (3+4), gripper 2 | `robocasa365` (finetuning only) |

The first 14 keys have a trained state MLP in `kyutai/reki-2b`. Finetuning adds
`panda_omron` and the mirrored keys `single_arm_ee_euler_mirrored`, `aloha_mirrored` and
`panda_omron_mirrored`, which have the format of their base key.

### Slices

- **`state_slice`** reorders or trims the stored state to the key layout. `None` inserts a zero
  pad. Example: `libero` stores 8 dims `[xyz, axis-angle, gripper_qpos(2)]`. The slice
  `[0, 1, 2, 3, 4, 5, None, 6]` keeps xyz and the rotation, inserts the pad and keeps the first
  gripper dim. The loader then converts axis-angle to euler (`unify_ee_euler_rotation`): the
  `single_arm_ee_euler` layout.
- **`action_slice`** drops only the DoF that the robot cannot or does not command: planar base
  roll and pitch, redundant end-effector dims, parked head or legs. Do not drop an axis only
  because it is constant in one dataset.
- Drop derived dims from the state: end-effector poses from forward kinematics, velocities,
  efforts, IMU. Keep dims that are constant within an episode but vary across episodes: they
  are real context.
- The same robot with fewer DoF: map it to the larger layout and zero-pad the missing dims.
- Find grippers by feature name (`*_gripper_open`), not by range.

### Control modes and prompt

`DatasetSpec.control_mode` is one of:

| Control mode | Datasets |
|---|---|
| `delta end-effector` | end-effector deltas: LIBERO, RoboCasa365 |
| `absolute joint` | joint targets: RoboTwin |

The robot identity reaches the model in a structured prompt, one `field: value` per line:
`embodiment`, `instruction`, `speed`, `fps`, `control mode`. `speed` is the episode length in 100-frame bins. Evaluation leaves it out
(`rollout.py --speed_bin N` sets it). The `embodiment` line is the dataset's
`DatasetSpec.robot_type`.

### Chunk size

`chunk_size = round(chunk_duration_s × fps)` at training and evaluation. The released configs use
1 s, so the chunk has `fps` steps. Each `predict` executes `history_stride_s × fps` of them
([predict cadence](architecture.md#inference)).

| Benchmark | fps | Chunk | Executed steps (K=3, 1 s stride) |
|---|---|---|---|
| LIBERO | 10 | 10 | 10 |
| RoboTwin 2.0 | 15 | 15 | 15 |
| RoboCasa365 | 20 | 20 | 20 |

## Normalization

**Quantiles.** Actions and state are normalized to [−1, 1] with their 1st and 99th percentiles.
The percentiles are computed over the whole dataset, so that the same pose maps to the same
value in every episode and task (`scripts/data/augment_quantiles.py` writes them as
`qk01`/`qk99` in `meta/stats.json`). A collection with an absolute control mode needs the same
`qk01`/`qk99` in every leaf (`augment_quantiles.py --leaves <collection root> --global`).

**Rotations.** With `unify_ee_euler_rotation: true`, the loader converts axis-angle rotations to
euler xyz (`DatasetSpec.action_rot_rep`, `state_rot_rep`) and normalizes them with
`meta/norm_stats_euler.json` (`scripts/data/recompute_euler_stats.py`). Evaluation converts the
predicted rotation back to axis-angle (single-arm robots only). The loader and the LIBERO
evaluation shift the LIBERO state roll by π, because the
gripper points down and its roll wraps at ±π.

**Grippers.** In every dataset, a high gripper value means open. `DatasetSpec.gripper_flip_state`
and `gripper_flip_action` negate the listed dims after normalization, and inference applies the
same flips. `scripts/data/check_datasets.py` checks the flips against the raw data.

## Mirroring

Mirroring turns a sample into its left-right mirror image: a mirror-image robot in a
mirror-image scene. The images flip horizontally, and the state and action get the same
reflection. Every finetuning run except LIBERO mirrors half of its samples (`mirror_prob: 0.5`). On a
bimanual robot, the arms and their wrist cameras swap. The instruction stays as it is, with a `[MIRROR WORLD] ` prefix.
Mirrored samples have their own state MLP (`<embodiment>_mirrored`) and their own
normalization statistics, mirrored from those of the dataset. Inference does not mirror.

To mirror a new robot, add a `MirrorSpec` in
[`reki/data/robotics/mirror.py`](../reki/data/robotics/mirror.py) and name it in
`DatasetSpec.mirror`. Check on your data which axis is left-right.

**Signs.** A `MirrorSpec` permutes and negates the raw state and action, before
`state_slice` and `action_slice` select the dims the model sees. With the mirror plane normal
to y (`M = diag(1, -1, 1)`), positions and translation deltas are vectors and take the sign
`[+1, -1, +1]`. Rotations are pseudovectors and take the opposite sign, `[-1, +1, -1]`; this
holds for axis-angle and euler rotations, wrapped or not, in the base or the end-effector
frame. A quaternion (xyzw) becomes `(-x, y, -z, w)`. A revolute joint keeps its angle when its
axis is parallel to the plane normal, and negates it otherwise. The two finger joints of the
robosuite Panda have opposite signs, so they swap and negate (`[q0, q1] → [-q1, -q0]`); a
scalar gripper value keeps its sign.

## Benchmark datasets

These steps build the finetuning datasets of the benchmarks under `$REKI_DATASETS_ROOT`, at
the paths the loader expects. LIBERO-Plus reuses the LIBERO finetune and needs no data. Run the
commands from the repository root with `pixi run python …`.

`scripts/data/download.py` fetches the raw inputs of the benchmark datasets.

| Dataset | Steps |
|---|---|
| `libero` | `scripts/data/download.py libero` (already v3.0), then `scripts/data/recompute_euler_stats.py --datasets libero` |
| `robotwin2_clean`, `robotwin2_randomized` | download the LeRobot-converted RoboTwin 2.0 demos (v2.1, one repo per task) of StarVLA: `pixi run hf download StarVLA/RoboTwin-Clean --repo-type dataset --local-dir $REKI_DATASETS_ROOT/robotwin2_clean_lerobot` and `StarVLA/RoboTwin-Randomized` with `--include 'Randomized/*' --local-dir $REKI_DATASETS_ROOT/robotwin2_randomized_lerobot`, then `scripts/data/prepare_robotwin.py --collections robotwin2_clean` (and `robotwin2_randomized`), or `COLLECTION=robotwin2_clean sbatch scripts/data/sbatch/prepare_robotwin.sbatch` |
| `robocasa365` | `scripts/data/download.py robocasa365 --manifest <robocasa checkout>/robocasa/models/assets/box_links/box_links_ds.json --only pretrain/atomic pretrain/composite`, then `scripts/data/prepare_robocasa365.py`. This gives Human300, the human demonstrations of the 300 pretraining tasks |

## Onboarding a dataset

Use these steps to add your own robot data.

1. **Convert to v3.0.** `scripts/data/build_v30.py --input <v2.x root> --output <dir>`
   rebuilds `meta/` only (`--link_data` symlinks the parquets).
2. **Your own recordings.** Write them with `LeRobotDataset.create(...)`, `add_frame`,
   `save_episode`, then **`finalize()`**.
   - Make one dataset per task directory.
   - Remove NaN values first.
3. **Quantiles.** Run `scripts/data/augment_quantiles.py --leaf <dir>`, or
   `--leaves <collection root>` for one repo per task. Add `--global` for an absolute control
   mode ([Normalization](#normalization)).
4. **Register.** Add one `DatasetSpec` to `DATASETS` in
   [`reki/data/robotics/registry.py`](../reki/data/robotics/registry.py):
   - `family`, the source grouped in the sampling-mix printout;
   - `robot_type`, the robot name in the prompt;
   - `control_mode` ([Control modes and prompt](#control-modes-and-prompt)) and `cameras`
     (raw key → `camera_top` / `camera_wrist_left` / `camera_wrist_right` / `camera_side_N`);
   - `embodiment`, with its dim in `state_mlp_embodiments` of your run;
   - `state_slice`, `action_slice`;
   - `gripper_flip_state` / `gripper_flip_action` (indices after the slice) if high means closed;
   - `action_rot_rep` / `state_rot_rep` for axis-angle rotations, then
     `scripts/data/recompute_euler_stats.py --datasets <name>`;
   - `subtask_lookup=True` if the leaves have a subtask lookup (checked with `predict_subtask`);
   - `root` for one repo, or `collection` for one repo per task; a group of datasets goes in
     `DATASET_GROUPS`.
5. **Validation split (optional).** `scripts/data/make_val_split.py --dataset <name>
   --per_task N` writes evenly spaced episodes to `meta/dropped_episodes.json` (the file means
   "not trained on"). Pin the same episode ids in `VAL_DATASETS`.
6. **Check.**
   - `scripts/data/check_datasets.py` on your data: group conventions, centered delta actions,
     per-step rotation below π, raw layout vs registry, and gripper polarity.
   - `scripts/dry_run.py --config=configs/sft.yaml --dataset=<name>`.
