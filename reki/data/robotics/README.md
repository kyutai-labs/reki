# Robot data

The training data pipeline: registered LeRobot v3.0 datasets → `CombinedLeRobotDataset` and a
weighted sampler → packed model batches. [docs/data.md](../../../docs/data.md) gives the
conventions; the scripts in [`scripts/data`](../../../scripts/data) prepare the datasets.

Start with `registry.py`: one `DatasetSpec` per dataset in `DATASETS`, and the groups in
`DATASET_GROUPS`. Adding a dataset is one `DATASETS` entry
([Onboarding a dataset](../../../docs/data.md#onboarding-a-dataset)). `TrainConfig.dataset` is a
dataset or group name, a list of them, or a comma-joined string (`expand_spec`).

## Training path

`factory.make_combined_dataset` builds the dataset. `RatioSampler` picks a leaf by its ratio, then
a frame; `CombinedLeRobotDataset` builds the sample. Each sample is mirrored (optional), sliced,
converted to euler, normalized, augmented and tokenized; `collate.py` packs a batch.

With `--external_policy`, the items hold unnormalized state and action and the decoded images
after the adapter's `preprocess_images`, and the adapter's `collate` replaces `collate.py`.

## Inference

Inference featurizes one observation with the same functions, without the dataset:
`reki.inference.features` uses `normalize_feature`, `negate_gripper_dims` and the `prompt.py`
functions; `reki.inference.spec` builds the shipped stats with `prepare_leaf_stats` and
`slice_leaf_stats`; the LeRobot plugin applies the state transform with `gather_with_pad`,
`axisangle_to_euler` and `recenter_euler_roll`.
