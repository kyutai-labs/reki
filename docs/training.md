# Training

| Config | Use |
|---|---|
| [`configs/sft.yaml`](../configs/sft.yaml) | finetuning config, shared by all finetuning runs, initialized from `kyutai/reki-2b` |
| [`configs/cpt.yaml`](../configs/cpt.yaml) | pretraining config of `kyutai/reki-2b`, initialized from `Qwen/Qwen3.5-2B` |

[`launchers/sft.py`](../launchers/sft.py), [`sft_3n.py`](../launchers/sft_3n.py) and
[`sft_4n.py`](../launchers/sft_4n.py) add the settings of each finetuning run. Every other
setting takes its default from [`reki/policy/train_config.py`](../reki/policy/train_config.py).

## Run

All commands run from the repository root.

```bash
# one batch, forward and backward, on one dataset: check the data and the model
pixi run python scripts/dry_run.py --config=configs/sft.yaml --dataset=libero --batch_size=4

# the LIBERO run, on all GPUs of this machine
pixi run train --config=configs/sft.yaml --pretrained_path=kyutai/reki-2b --dataset=libero \
    --steps=15000 --warmup_steps=1000 --mirror_prob=0.0 \
    --state_mlp_embodiments="{'single_arm_ee_euler': 8}" \
    --aug_random_crop=0.95 --aug_random_scale=True --pin_overhead_camera=False \
    --output_dir=runs/libero
```

- Command-line flags override the config (`--steps=100`). `--output_dir` is required.
- The flags after `--dataset` are the LIBERO settings of `launchers/sft.py`. To print the full
  command of another run: `pixi run python launchers/<launcher>.py select <i> cmd`.
- `--resume=True` continues from the latest `checkpoint-N` in `output_dir`.
- `pixi run train` is `accelerate launch scripts/train.py`.
- Every `logging_steps`, training writes the loss, the learning rate, the gradient norm, the
  step and data-loading times, the batch size in samples and tokens, the AdaRMS gates and the
  peak GPU memory to stdout and `<output_dir>/metrics.jsonl`. The gates (`adarms_gate_mean`)
  start at zero and should grow during training.
- Every 5,000 steps (`save_steps`), training saves a checkpoint: `model.safetensors`,
  `ema.safetensors` and `train_config.yaml`.

### Slurm

The launchers submit to Slurm through [launcheon](https://github.com/kyutai-labs/launcheon).
[`launchers/common.py`](../launchers/common.py) reads the cluster settings from environment
variables. Set them in `.env` ([README](../README.md#installation)):

| Variable | Meaning |
|---|---|
| `REKI_RUNS_DIR` | root of the run directories (default: `storage/runs` in the repository). `<root>/<grid>/<hash>/` holds the Slurm script, the log, the checkpoints and the eval results. |
| `SBATCH_PARTITION` | the partition of all jobs (default: the cluster default). `submit --partition <name>` overrides it for one command |
| `REKI_EVAL_MEM_PER_GPU` | host RAM of an eval job, e.g. `240G` (default: the Slurm default). Set it when your cluster gives a job without a memory request the whole node |
| `REKI_GPUS_PER_NODE` | GPUs per node (default 8) |
| `REKI_CPUS_PER_GPU` | CPUs per GPU (default 16) |
| `REKI_CONTAINER_IMAGE`, `REKI_CONTAINER_MOUNTS` | pyxis container of the jobs |
| `HF_DATASETS_CACHE` | Hugging Face `datasets` cache of the training data (default: `$HF_HOME/datasets`). It needs a filesystem with working file locks |

```bash
pixi run python launchers/sft.py table                               # one row per finetuning run
pixi run python launchers/sft.py select 0 submit                     # finetune on LIBERO (8 GPUs)
pixi run python launchers/sft.py select 0 submit --partition gpu     # on another partition
pixi run python launchers/sft.py select 0 cmd                        # print the training command
pixi run python launchers/sft.py select 0 print log -20              # last 20 lines of the log
pixi run python launchers/sft.py select 0 eval --task libero submit  # evaluate the latest checkpoint
pixi run python launchers/sft.py select 0 results                    # success rate per checkpoint
```

- The jobs run with `--resume=True`, so a resubmitted job continues the run.
- `submit` downloads the models into the Hugging Face cache first. The compute nodes must see
  `HF_HOME`.
- To finetune from your own pretraining run, set `PRETRAINED` in `launchers/sft.py`.
- [`launchers/sft_scratch.py`](../launchers/sft_scratch.py) runs LIBERO and RoboTwin clean
  without pretraining: from `Qwen/Qwen3.5-2B`, with a 5,000-step warmup and 30k / 40k steps.
- [evaluation.md](evaluation.md) describes the `eval` and `results` verbs.

## Training setup

**Loss.** The action chunk is 1 s long ([chunk size](data.md#chunk-size)). The flow-matching loss
averages 8 draws of the noise and the flow time per frame (`flow_draws`,
[architecture.md](architecture.md#training)) and gives each sample the same weight, whatever its
number of action dimensions (`flow_loss_dof_balanced`). In pretraining, the VLM also learned to predict the subtask and the
answers of the vision-language data, with a cross-entropy weight of 0.1. Finetuning trains only
the flow-matching loss.

**Optimizer and precision.** The model trains in bf16, with the flow heads in fp32. The optimizer
is AdamW with betas (0.9, 0.95) and fp32 master weights. The learning rate depends on the part of
the model:

| Parameters | Learning rate |
|---|---|
| action expert (layers and register tokens), state MLPs | 1e-4 (`lr`) |
| VLM text stack, LM head, action and time projections | `lr × backbone_lr_scale`: 0.1 in pretraining, 0.3 in finetuning ([per run](#finetuning-runs)) |
| last 4 vision blocks (`train_vision_blocks: 4`) | `lr × backbone_lr_scale` |
| other vision blocks | frozen |

The schedule warms up linearly, stays constant, then decays with a cosine to 2e-6 over the last
`decay_steps` (`lr_schedule: wsd`). Pretraining does not decay. Finetuning decays over the last
5,000 steps. Weight decay is 0 in pretraining and 1e-8 in finetuning.
Gradients are clipped at 1.0. The evaluation and the released models use the EMA weights
(decay 0.999).

**Augmentation.**

- **Images:** color jitter of 0.2 on every image. In pretraining and on LIBERO, also a random
  crop of the overhead and side cameras to 95% of the image, with a random scale.
- **Cameras:** each camera is dropped with probability 0.2, never all of them, and the camera
  order is shuffled. The overhead camera is never dropped, except on LIBERO.
- **Mirroring:** in the RoboTwin and RoboCasa365 runs, half of the samples are
  mirrored left to right ([data.md](data.md#mirroring)). They get their own state MLP and the
  `[MIRROR WORLD]` prefix on the instruction.
- **Prompt:** the embodiment, control-frequency and speed lines are dropped with probability
  0.15, 0.2 and 0.2.
- **State:** a learned null state replaces the robot state in 20% of the samples
  (`state_dropout_prob`). Datasets without state, such as the human-hand videos, always use it.

**Sampling.** The sampler picks a dataset by its sampling probability, then a random frame of
that dataset. This frame is the last frame of the window. In finetuning, the probability of a dataset is proportional to the square root of its
frame count (`frames_alpha: 0.5`). The pretraining probabilities are in
[data.md](data.md#pretraining-mixture). The probabilities are then capped so that no dataset is
seen more than `max_epochs_per_dataset` times (4 in pretraining, 100 in finetuning).

**Batch size.** `batch_size` (30) is an upper bound per GPU. Packing drops whole windows from the
end until the tokens fit in `max_text_tokens` ([architecture.md](architecture.md#training)). The batch
size of an optimizer step is these windows × GPUs × gradient accumulation
([finetuning runs](#finetuning-runs)).
Gradient checkpointing (`gradient_checkpointing: true`) recomputes the activations of the VLM
and action-expert layers in the backward pass.

## Finetuning runs

All runs start from `kyutai/reki-2b`.

| Run | Dataset | Steps | Warmup | GPUs × grad. accumulation | GPU-hours | Differences from `sft.yaml` |
|---|---|---|---|---|---|---|
| `libero` | `libero` | 15,000 | 1,000 | 8 × 1 | 40 | random crop, the overhead camera can drop |
| `robotwin_clean` | `robotwin2_clean` | 20,000 | 1,000 | 8 × 1 | 90 | mirroring, VLM rate ×0.2 |
| `robotwin_cleanrand` | `robotwin2_clean`, `robotwin2_randomized` | 25,000 | 1,000 | 24 × 1 | 340 | mirroring |
| `robocasa365` | `robocasa365` | 25,000 | 1,000 | 32 × 1 | 400 | mirroring |

[`sft.py`](../launchers/sft.py) runs on one node of 8 H100 GPUs, [`sft_3n.py`](../launchers/sft_3n.py)
on three nodes and [`sft_4n.py`](../launchers/sft_4n.py) on four. A GPU uses up to 73 GB of memory
(RoboTwin), so use 80 GB GPUs.
[data.md](data.md#benchmark-datasets) gives the steps to build the benchmark datasets.

### Finetune on your robot

1. Convert your data to LeRobot v3.0 and register it ([data.md](data.md#onboarding-a-dataset)).
2. Add an entry to `BENCHMARKS` in [`launchers/sft.py`](../launchers/sft.py): `dataset`,
   `steps`, `warmup_steps`, `mirror_prob` (0.0 unless you [register a mirror](data.md#mirroring) for the robot),
   `state_mlp_embodiments` (the state format of your robot and its dim) and
   `gradient_accumulation_steps`. The entry needs no evaluation task.
3. Train it with the [Slurm](#slurm) commands, or with `pixi run train` and the fields of the
   entry as flags.
4. Serve the checkpoint ([README](../README.md#real-robot-deployment)).

Our UR5 runs used 50 demonstrations per task and these settings (8 H100 GPU-hours):

- `steps=3500`, at a global batch of 360
- `lr=5e-5`, `backbone_lr_scale=0.1`, `decay_steps=500`
- `chunk_duration_s=2.0`: 2 s action chunks
- `history_stride_s=1.0`: the policy replans every second during asynchronous inference

### Export

`scripts/export_checkpoint.py` writes a training checkpoint in the layout of the released
models: weights (with the EMA merged), a cleaned `train_config.yaml` and `config.json`, and,
with `--dataset` or `--benchmark`, the inference spec and normalization statistics that
`RekiPolicy` needs without the training data.

```bash
pixi run python scripts/export_checkpoint.py runs/libero/checkpoint-15000 exports/libero \
    --merge-ema --pretrained-path kyutai/reki-2b --benchmark libero
pixi run python scripts/export_checkpoint.py runs/my_robot/checkpoint-30000 exports/my_robot \
    --merge-ema --pretrained-path kyutai/reki-2b --dataset <your_dataset>
```

## Config rules

**Format.** `configs/cpt.yaml` is the reference recipe of `kyutai/reki-2b`; the pretraining code that runs it comes in a later update. `dataset` is
a list, a group or dataset name, or a comma-joined string of names. Data paths in the configs are
relative to the repository. The datasets are read under `REKI_DATASETS_ROOT`.

**State MLPs.** `state_mlp_embodiments` lists the state formats of the run's data
([data.md](data.md#state-formats)). A format of the start checkpoint loads its trained MLP and
must keep its dim. A new format gets a new MLP, trained from scratch.
