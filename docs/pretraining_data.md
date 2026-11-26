# Pretraining data

These steps build the data of [`configs/cpt.yaml`](../configs/cpt.yaml), with its normalization
statistics and lists of excluded episodes, under `$REKI_DATASETS_ROOT`.

## Pretraining mixture

The mixture of [`configs/cpt.yaml`](../configs/cpt.yaml) holds 894M frames, without the
excluded episodes: real-robot teleoperation (569M), simulation (177M) and egocentric human video
(148M), plus 12.2M vision-language rows. The four parts are sampled with probabilities of
52.6%, 10.1%, 27.2% and 10.0%.

| Source | Robots | Frames | Sampling probability | License | Download | Built |
|---|---|---|---|---|---|---|
| [Open X-Embodiment](https://huggingface.co/IPEC-COMMUNITY)* | Google Robot, Franka | 14.6M | 4.7% | Apache-2.0 | 48 GB | 66 GB |
| [BridgeData V2](https://huggingface.co/datasets/Gaugou/BridgeV2)* | WidowX | 1.8M | 1.5% | CC BY 4.0 | 61 GB | 61 GB |
| [DROID 1.0.1](https://huggingface.co/datasets/lerobot/droid_1.0.1) | Franka | 23.4M | 6.4% | Apache-2.0 | 0.39 TB | 0.39 TB |
| [AgiBot World 2026](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026)* | AgiBot G2A | 17.4M | 4.3% | CC BY-NC-SA 4.0 | 2.9 TB | 2.1 TB |
| [Galaxea Open-World](https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset) | Galaxea R1 Lite | 18.6M | 6.6% | CC BY-NC-SA 4.0 | 2.9 TB | 0.8 TB |
| [RoboMIND 1.2](https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND)* | Franka, dual FR3, UR5e, AgileX, TienKung | 17.4M | 5.2% | Apache-2.0 | 7.5 TB | 0.15 TB |
| [rdt-ft-data](https://huggingface.co/datasets/robotics-diffusion-transformer/rdt-ft-data) | ALOHA-style bimanual | 3.2M | 2.2% | MIT | 0.70 TB | 18 GB |
| [RoboCOIN](https://huggingface.co/RoboCOIN)* | AgileX, R1 Lite, AIRBOT MMK2, Unitree G1, RealMan, Galbot, AgiBot A2D, Leju Kuavo, AI2 AlphaBot 2 | 99.3M | 10.3% | Apache-2.0 | 4.9 TB | 4.9 TB |
| [ABC-130K](https://huggingface.co/datasets/XDOF/ABC-130k)* | YAM bimanual | 373.0M | 11.1% | Apache-2.0 | 21.8 TB | 8.8 TB |
| [RoboSet-Teleoperation](https://huggingface.co/datasets/jdvakil/RoboSet-Teleoperation) | Franka | 0.2M | 0.1% | MIT | 0.43 TB | 16 GB |
| [GR00T X-Embodiment Sim](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim)* | GR-1 humanoid, bimanual Panda | 12.5M | 1.9% | CC BY 4.0 | 0.11 TB | 0.11 TB |
| [InternData-A1](https://huggingface.co/datasets/InternRobotics/InternData-A1)* | Franka, ALOHA, Genie-1 | 164.4M | 8.3% | CC BY-NC-SA 4.0 | 0.92 TB | 0.9 TB |
| [EgoDex](https://github.com/apple/ml-egodex)* | human hands | 88.4M | 16.7% | CC BY-NC-ND 4.0 | 1.86 TB | 1.64 TB |
| [EgoVerse](https://github.com/GaTech-RL2/EgoVerse)* | human hands | 11.7M | 2.9% | CC BY-SA 4.0 | 37 GB | 30 GB |
| [Xperience-10M](https://huggingface.co/datasets/ropedia-ai/xperience-10m)* | human hands | 48.1M | 7.7% | Xperience license | 0.34 TB | 186 GB |
| [FineVision](https://huggingface.co/datasets/HuggingFaceM4/FineVision)* | — | 6.3M rows | 5.2% | per sub-dataset | 2.04 TB | 2.06 TB |
| [LLaVA-OneVision-1.5](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Instruct-Data)* | — | 1.3M rows | 1.0% | Apache-2.0 | 0.51 TB | 0.51 TB |
| [RoboPoint](https://huggingface.co/datasets/wentao-yuan/robopoint-data)* | — | 0.6M rows | 0.5% | Apache-2.0 | 0.22 TB | 0.20 TB |
| [RefSpatial](https://huggingface.co/datasets/JingkunAn/RefSpatial)* | — | 2.3M rows | 1.9% | Apache-2.0 | 0.29 TB | 0.55 TB |
| [PixMo-Points](https://huggingface.co/datasets/allenai/pixmo-points) | — | 1.7M rows | 1.4% | ODC-BY | 0.39 TB | 0.39 TB |

\* Only part of the source is used:

<details>
<summary>Parts used</summary>

- **Open X-Embodiment:** 12 LeRobot exports `IPEC-COMMUNITY/<set>_lerobot`: `taco_play`,
  `stanford_hydra_dataset`, `nyu_franka_play_dataset`, `bc_z`, `fractal20220817_data` (RT-1),
  `furniture_bench_dataset`, `austin_sailor_dataset`, `austin_buds_dataset`, `utaustin_mutex`,
  `berkeley_cable_routing`, `viola`, `berkeley_rpt`.
- **BridgeData V2:** the teleoperated demonstrations (`demos_8_17`), without the scripted ones,
  from the LeRobot v3.0 conversion `Gaugou/BridgeV2`.
- **AgiBot World 2026:** `ImitationLearning/`, without `simulation/`.
- **RoboMIND 1.2:** the `benchmark1_*_compressed` folders, without `h5_simulation` and `h5_sim_*`.
- **RoboCOIN:** 672 task repos `RoboCOIN/<task>` (snapshot of 2026-06-04); 19 other `R1_Lite_*`
  repos, whose state is not 14-dim, are not used.
- **ABC-130K:** `data/train/`, without `assemble_paper_box`,
  `insert_wireless_bluetooth_earbuds_into_charging_case`,
  `pack_books_pencil_case_and_school_supplies_into_a_backpack`, `unlock_a_padlock_with_a_key`.
- **GR00T X-Embodiment Sim:** 11 tasks: `gr1_arms_only.CanSort`;
  `gr1_arms_waist.{CanToDrawer, CupToDrawer, CuttingboardToBasket, CuttingboardToCardboardBox}`;
  `bimanual_panda_gripper.{Threading, ThreePieceAssembly, Transport}`;
  `bimanual_panda_hand.{BoxCleanup, DrawerCleanup, LiftTray}`.
- **InternData-A1:** `sim_updated_lerobotv30/`, without `pick_and_place_tasks/`, `lift2/` and
  `split_aloha/make_a_sandwich_language`.
- **EgoDex:** all, without `test`.
- **EgoVerse:** the 2,514 Aria episodes of `processed_v3/aria/`: the `<episode>.mp4` videos, the
  head poses and the hand keypoints; not the JPEG frames.
- **Xperience-10M:** all, without the `fisheye_cam*` and `stereo_right` videos.
- **FineVision:** 51 configs, selected by hand ([Vision-language data](#vision-language-data)).
- **LLaVA-OneVision-1.5:** `gqa`, `llava_cot_100k`, `llava_wild`, `llrv_gpt4v`, `sherlock`,
  `svit-part-*`, `viquae`, `vision_oritented`, `visual_chat`.
- **RoboPoint:** `object_ref` and `region_ref`; not `VG_100K`, `coco`, `gqa`, `ocr_vqa`,
  `text_vqa`.
- **RefSpatial:** `2D/` and `3D/`, without their `depth*` folders.

</details>

## Setup

- Linux x86-64, with REKI installed as in the [README](../README.md#installation).
- The `data` environment: `pixi install -e data --frozen`.
- The datasets root, in `.env` or in the shell: `export REKI_DATASETS_ROOT=/path/to/datasets`.

### Access

Accept the terms on the Hugging Face page of RoboMIND, Galaxea Open-World Dataset,
InternData-A1, ABC-130K and each gated RoboCOIN repository.

Xperience-10M needs an access request and a signed agreement on its page.

EgoVerse needs an access request to its team ([EgoVerse](https://github.com/GaTech-RL2/EgoVerse)).
Set its storage credentials in `R2_ENDPOINT_URL`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` and
`R2_BUCKET`, in `.env` or in the shell.

EgoDex and EgoVerse need the MANO hand model: register at
[mano.is.tue.mpg.de](https://mano.is.tue.mpg.de), download `mano_v1_2.zip` and put it in
`upstream/mano/` of the datasets root.
The 10 finger eigengrasps of the human-hand actions use a fixed PCA basis,
`scripts/data/pretrain/readers/assets/eigengrasp_pca.npz`, fit on the MANO finger poses of
VITRA, EgoVerse, EgoDex and Xperience-10M.

## Build

Each group is built in five stages:

| | Stage | Role | Output |
|---|---|---|---|
| 1 | `download` | the pinned upstream files are downloaded | `$REKI_DATASETS_ROOT/upstream/` |
| 2 | `build` | the upstream files are converted to LeRobot datasets | the data, videos and `meta/` of each dataset |
| 3 | `curate` | the listed episodes of each dataset are excluded from training | `meta/dropped_episodes.json`; subtask labels for AgiBot World 2026, Galaxea, ABC-130K and Xperience-10M |
| 4 | `stats` | the normalization statistics are computed | `meta/stats.json` and the other statistics files |
| 5 | `verify` | the counts, files and one training sample are checked | — |

```bash
pixi run -e data python scripts/data/pretrain_data.py download <name>
pixi run -e data python scripts/data/pretrain_data.py build <name>
pixi run -e data python scripts/data/pretrain_data.py curate <name>
pixi run -e data python scripts/data/pretrain_data.py stats <name> --config configs/cpt.yaml
pixi run -e data python scripts/data/pretrain_data.py verify <name> --config configs/cpt.yaml
```

`<name>` is a group or a dataset of the table below.

`robocoin`, `interndata`, `abc`, `egodex`, `egoverse` and `xperience` are large: split their
`build` into `n` parallel jobs with `--shard i/n` (`i` = 0 … n−1).

| Group | Datasets | Output under `$REKI_DATASETS_ROOT` |
|---|---|---|
| `oxe_reduced` | `taco_play`, `stanford_hydra_dataset_converted_externally_to_rlds`, `nyu_franka_play_dataset_converted_externally_to_rlds`, `bc_z`, `fractal20220817_data`, `furniture_bench_dataset_converted_externally_to_rlds`, `austin_sailor_dataset_converted_externally_to_rlds`, `austin_buds_dataset_converted_externally_to_rlds`, `utaustin_mutex`, `berkeley_cable_routing`, `viola`, `bridge_v2`, `berkeley_rpt_delta` | `open_x_embodiment_lerobot/` |
| `droid` | `droid` | `open_x_embodiment_lerobot/droid_v30_lerobot/` |
| `agibot_world_2026` | `agibot_world_2026` | `agibot_world_2026_lerobot_v30/` |
| `robomind` | `robomind_franka_1rgb`, `robomind_franka_3rgb`, `robomind_ur_1rgb`, `robomind_franka_fr3_dual`, `robomind_agilex_3rgb`, `robomind_tienkung_gello_1rgb`, `robomind_tienkung_prod1_gello_1rgb`, `robomind_tienkung_xsens_1rgb` | `robomind_lerobot/<dataset>_lerobot_v30/` |
| `galaxea` | `galaxea` | `galaxea_lerobot_v30/` |
| `roboset_delta` | `roboset_delta` | `roboset_delta_lerobot/` |
| `rdt` | `rdt` | `rdt_lerobot/` |
| `robocoin` | `robocoin_a2d_17`, `robocoin_a2d_41`, `robocoin_agilex_14`, `robocoin_agilex_26`, `robocoin_agilex_26b`, `robocoin_airbot_mmk2`, `robocoin_galaxea_r1lite`, `robocoin_galbot`, `robocoin_leju`, `robocoin_r1lite`, `robocoin_realman_schemaA`, `robocoin_realman_schemaB`, `robocoin_unitree_g1_28`, `robocoin_unitree_g1_30` | `robocoin_lerobot_v30_collections/<dataset>/` |
| `groot` | `groot_gr1`, `groot_panda_gripper`, `groot_panda_hand` | `groot_x_embodiment_sim_lerobot_v30_collections/` |
| `interndata` | `interndata_franka`, `interndata_split_aloha`, `interndata_genie1` | `interndata_a1_lerobot_v30_collections/` |
| `abc` | `abc_yam_realsense`, `abc_yam_zedx` | `abc_yam_realsense_lerobot/`, `abc_yam_zedx_lerobot/` |
| `egodex` | `egodex` | `egocentric_raw/` |
| `egoverse` | `egoverse` | `egocentric_raw/` |
| `xperience` | `xperience` | `egocentric_raw/` |
| `vl` | `finevision`, `llava_onevision`, `robopoint`, `refspatial`, `pixmo` | `vl/` ([Vision-language data](#vision-language-data)) |

The groups are the entries of `dataset:` in `configs/cpt.yaml`, plus `vl`. Each dataset is stored
as one or more LeRobot folders (leaves), usually one per upstream task or repo.

In `roboset_delta` and `berkeley_rpt_delta`, the joint actions are converted to end-effector
deltas.

### Resources

- **GPU:** ABC-130K, EgoDex and EgoVerse need GPUs to build: one GPU per EgoDex or EgoVerse
  shard; for ABC-130K, about 14 shards per GPU, with one shard per 2 CPU cores.
- **RoboMIND 1.2:** give `build` all the CPU cores of the node, and up to 0.4 TB for `.work/`.
- **RoboPoint, RefSpatial:** give `build` about 32 GB of RAM.

### After the build

After `verify`, you can delete `$REKI_DATASETS_ROOT/upstream/` and `$REKI_DATASETS_ROOT/.work/`.

### Vision-language data

The entry `finevision_for_robotics` of `vl_datasets` in `configs/cpt.yaml` holds `finevision` and
`llava_onevision`. Only `download`, `build` and `verify` apply to the `vl` group.

| Dataset | Configs | Rows |
|---|---|---|
| `finevision` | the 51 configs of `FINEVISION_FOR_ROBOTICS` (`reki/data/robotics/vl.py`) | 6,349,586 |
| `llava_onevision` | the 9 configs of `LLAVA_EXT_FOR_ROBOTICS` | 1,291,426 |
| `robopoint` | `robopoint_object_ref`, `robopoint_region_ref` | 556,130 |
| `refspatial` | the 7 2D and 3D QA sets (`refspatial_2d_choice`, …) | 2,338,803 |
| `pixmo` | `pixmo_points` | about 1.66M |

If you move `vl/finevision/`, `vl/llava_onevision/` or `vl/spatial/` after the build, set
`FINEVISION_PATH`, `LLAVA_EXT_PATH` or `VL_SPATIAL_PARQUET_PATH` in `.env` to its new place.

**PixMo-Points.** The images are downloaded from their source sites, so some can be missing or
changed. To retry the failed images, run `download` again.

## Check

After the build of one dataset, run one forward and backward pass on one GPU:

```bash
pixi run python scripts/dry_run.py --config=configs/cpt.yaml --dataset=<name>
```

`<name>` is a group or a dataset of the Build table, except `vl`. To include vision-language data,
add `--vl_datasets=<entry>`, where `<entry>` is an entry of `vl_datasets` in `configs/cpt.yaml`.
To check a `--subset` build, run this command with `REKI_DATASETS_ROOT=<the --root of the subset>`.

To check the full config, first build every group of the [Build](#build) table, `vl` included,
and the `libero` validation data ([data.md](data.md#benchmark-datasets)). Then run:

```bash
pixi run python scripts/dry_run.py --config=configs/cpt.yaml
```

## Reference

| Flag | Meaning |
|---|---|
| `--root R` | the datasets root (default `$REKI_DATASETS_ROOT`) |
| `--subset` | a small part of each dataset, to check access and setup. Use a separate `--root` with it |
| `--shard i/n` | `build` only: part `i` of `n` of each dataset. You can use it with `rdt`, `roboset_delta`, `egodex`, `egoverse`, `xperience` and the `robocoin`, `groot`, `interndata`, `abc` and `vl` groups |
| `--config Y` | the training config of `stats` and `verify`; required for `egodex`, `egoverse` and `xperience` |
| `--detect` | `curate` only: make the lists of excluded episodes again (see below) |

`scripts/data/pretrain/drops/<dataset>.json` lists the episodes of a dataset that are excluded from
training: duplicates, episodes where the robot does not move, episodes with sudden jumps in the
state or action, and DROID episodes labeled as not successful.

<details>
<summary>Excluded episodes</summary>

If you change how a dataset is read, or a rule in this table, run `curate --detect` to make these
lists again.

| Reason | Datasets | Rule |
|---|---|---|
| `duplicate` | all | episodes of one dataset with the same state, action and length. One copy is kept: the first, or, if the task texts differ, the copy whose text matches the video |
| `frozen` | most real-robot datasets | the robot state does not move |
| `sudden_change` | InternData | a jump in the state or action |
| `ur_frozen_arm` | `robomind_ur_1rgb` | the six arm joints do not move |
| `failure` | `droid` | the collector labeled the episode as not successful |
| `video_cuts` | `droid` | ≥ 25% of the episodes of a top-camera video file contain a scene cut |
| `short_video` | robot datasets with upstream videos | a video holds fewer frames than the episode has rows |

</details>
