# Evaluation

REKI is evaluated in simulation on LIBERO, LIBERO-Plus, RoboTwin 2.0 and RoboCasa. This page
gives the protocol of each benchmark and the commands that reproduce the reported numbers. First install
the simulators: [scripts/eval/SETUP.md](../scripts/eval/SETUP.md).

## Benchmarks

| Task | Benchmark | Simulator |
|---|---|---|
| `libero` | LIBERO, 4 suites × 10 tasks | MuJoCo, CPU render, in the eval process |
| `libero_plus` | LIBERO-Plus, 4 suites × 7 perturbation axes | MuJoCo, CPU render, in the eval process |
| `robotwin` | RoboTwin 2.0, 50 tasks, Easy (`demo_clean`) and Hard (`demo_randomized`) | SAPIEN, GPU render (Vulkan), [server](#sim-servers) |
| `robotwin_easy`, `robotwin_hard` | `robotwin` with Easy or Hard only | same |
| `robotwin_clean2rand` | RoboTwin Clean2Rand: `demo_clean` with one of background, light, clutter or height randomized | same |
| `robocasa365` | RoboCasa365, the 50 target tasks | MuJoCo, CPU render, [server](#sim-servers) |

## Evaluate the released models

[`launchers/eval.py`](../launchers/eval.py) evaluates the finetuned
[released models](../README.md#released-models). It submits the jobs to Slurm and writes the
results under `$REKI_RUNS_DIR/models/`. `select N` picks the model in row N of `table`.

```bash
pixi run python launchers/eval.py table                                    # the models and their tasks
pixi run python launchers/eval.py select 0 eval --task libero submit       # reki-2b-libero on LIBERO
pixi run python launchers/eval.py select 0 eval --task libero_plus submit  # reki-2b-libero on LIBERO-Plus
pixi run python launchers/eval.py select 0 results                         # one table per task
```

To evaluate your own export, add a line to `MODELS` in `launchers/eval.py`.

## Evaluate your checkpoints

To evaluate a trained model, use the `eval` command of its training launcher
([`launchers/sft.py`](../launchers/sft.py), `sft_3n.py` or `sft_4n.py`).
As for the released models, `select N` picks the run in row N of `table`.

```bash
pixi run python launchers/sft.py select 0 eval --task libero submit               # latest checkpoint
pixi run python launchers/sft.py select 0 eval --task libero --step 15000 submit  # one checkpoint
pixi run python launchers/sft.py select 1 eval --task robotwin_clean2rand submit
pixi run python launchers/sft.py select 0 results                                 # one row per checkpoint
```

By default, `eval` uses the EMA weights of the latest checkpoint. The results go to the run
directory, next to the checkpoints. All reported numbers use the EMA weights.

Without Slurm, `launchers/eval_protocols.py` prints the same evaluation commands for one
checkpoint. Run them one after another on one GPU, then print the success rates:

```bash
pixi run python launchers/eval_protocols.py <task> <checkpoint> <out_dir> | bash
pixi run python scripts/eval/results_table.py <out_dir>
```

The checkpoint is a training `checkpoint-N/` or an exported checkpoint. A training
checkpoint uses its EMA weights (`--ema=False`: the raw weights), and it reads its normalization
statistics from the training dataset under `REKI_DATASETS_ROOT`. An exported checkpoint (an HF
model id such as `kyutai/reki-2b-libero`, or the output of `scripts/export_checkpoint.py
--benchmark`, see [training.md](training.md#export)) holds its normalization statistics and the
EMA weights.

| Option | Meaning |
|---|---|
| `--n_episodes N` | episodes per task ([Protocol](#protocol); default: the protocol) |
| `--num_workers K` | rollout processes per job (default 6). RoboTwin, RoboCasa: also the tasks per job, one worker per task |
| `--tasks a,b` | only these LIBERO suites, or these RoboTwin or RoboCasa tasks |
| `--difficulties a,b` | only these RoboTwin scenes (`easy`, `hard`) or Clean2Rand randomizations |
| `--step S` | `eval` only: `20000`, `20000,25000`, `15000-25000` or `all` (default: the latest checkpoint) |
| `--task T` | `eval` only: a task of [Benchmarks](#benchmarks) |
| `--resume=True` | continue an earlier run ([Results](#results)) |
| `--save_videos=True` | one video per episode ([Results](#results)) |
| `--history_k K` | frames in the window (default: the trained K). The trained K writes the plain results. To see other K: `results --history_k K` or `results --history_k 1,2,3` (one row per K) |
| `--n_action_steps N` | actions executed per predict (default: 1 s of actions, then replan; a K>1 model always uses 1 s). To see these results: `results --n_action_steps N` |
| `--sim_port`, `--sim_host`, `--sim_relay=True` | RoboTwin servers that already run; opt in to a relay from the submit node (see [RoboTwin](#robotwin-robotwin_clean2rand)) |

Write the yes/no options (`--ema`, `--resume`, `--save_videos`, `--sim_relay`) as `=True` or
`=False`.

Each benchmark needs its [environment variables](#environment-variables).

## Protocol

Each job runs `scripts/eval/rollout.py` with the settings below. To print the exact commands:

```bash
pixi run python launchers/eval_protocols.py <task> <ckpt_dir> <out_dir>
```

LIBERO runs one job per suite, and LIBERO-Plus one job per suite and axis. RoboTwin and RoboCasa
run one job per `--num_workers` tasks (RoboCasa365: within one suite).

| Benchmark | Episodes per task | Executed steps per `predict` | Episode limit (steps) | Tasks |
|---|---|---|---|---|
| LIBERO | 50 | 10 (10 Hz, 1 s stride) | spatial 280, object 280, goal 300, `libero_10` 520 (LeRobot's `LiberoEnv`) | 4 suites × 10 tasks |
| LIBERO-Plus | 1 | 10 | as LIBERO | 4 suites × 7 axes, 10,030 perturbed tasks |
| RoboTwin Easy / Hard | 20 | 15 (15 Hz, 1 s stride) | per task (RoboTwin's `_eval_step_limit.yml`) | 50 tasks |
| RoboTwin Clean2Rand | 20 | 15 | as Easy / Hard | 50 tasks × {background, light, clutter, height} |
| RoboCasa365 | 50 | 20 | per task, 450 to 4350 (`robocasa`'s `get_task_horizon`) | 50 target tasks in the pretraining kitchens |

- **Seed.** `--seed 42` (the default). Each task draws its episode seeds from
  this seed.
- **Window.** `--history_k` is not passed: the trained window (K=3) is used.
- **Cadence.** `--n_action_steps` is not passed: 1 s of actions, then replan
  ([predict cadence](architecture.md#inference)).
- **Compiled inference.** `--compile` is on by default. The compiled model is cached in
  `--compile_cache_dir` (default `~/.cache/reki/inductor`).
- **LIBERO-Plus instruction bug.** Most LIBERO-Plus instructions incorrectly end with the
  perturbation parameters, for example `… view 0 0 100 0 0 initstate 1`. LIBERO-Plus makes these
  instructions from the task file name, and the fix is not merged upstream
  ([LIBERO-plus#65](https://github.com/sylvestf/LIBERO-plus/issues/65)). We keep these
  instructions for comparability with prior work.

## Metrics

| Benchmark | Reported number |
|---|---|
| LIBERO | mean over the 40 tasks (equal to the mean of the 4 suites) |
| LIBERO-Plus | successes over all episodes, overall and per axis |
| RoboTwin | mean over the tasks |
| RoboTwin Clean2Rand | mean of 6 scene numbers: Easy and Hard (`robotwin`), and the four randomizations (`robotwin_clean2rand`) |
| RoboCasa365 | mean over the 50 target tasks |

## Results

```
<out_dir>/libero/<suite>/<task_id>/results.json
<out_dir>/libero_plus/libero_plus_<suite>/<axis>/<task_id>/results.json
<out_dir>/robotwin_<easy|hard|background|light|clutter|height>/<task>/results.json
<out_dir>/robocasa365/<Task>/results.json
```

- Each `results.json` holds the episodes of one task (`seed`, `steps`, `success`, `terminal`) and
  the run config.
- `--resume=True` skips the complete tasks and finishes the others with the same seeds.
- `launchers/sft.py select N results` (or `launchers/eval.py`) prints, per run and checkpoint,
  `trials` (the episodes done and expected), `steps` (the mean env steps of the successful
  episodes) and the success rates. A grey number has missing episodes or tasks.
  `results --task robotwin_clean2rand` also shows Easy and Hard, and the mean of the 6 scenes
  in bold: the Clean2Rand number.
- Without the launchers, `scripts/eval/results_table.py <out_dir> [<out_dir> ...] [--json]`
  prints the success rate of each group (suite, axis or task) and the benchmark number.
  `partial` marks a group with missing episodes or tasks.
- `--save_videos=True` writes one video per episode. To watch them, run
  `pixi run python scripts/eval/viewer.py <out_dir> --port 8000` and open
  `http://localhost:8000`.
- `scripts/eval/rollout.py --save_traj=True` writes `traj/epNNN.npz` per episode: the predicted
  chunks, the executed actions, the states and the sim state (except RoboTwin).

## Environment variables

Set them in `.env` at the repository root ([`.env.example`](../.env.example)). A variable set in
the shell overrides the file.

### All benchmarks

| Variable | Value |
|---|---|
| `REKI_SIM_ROOT` | the simulators, one directory per benchmark (default: `storage/sim` in the repository). `bash scripts/eval/install.sh <benchmark>` installs one ([SETUP](../scripts/eval/SETUP.md)). |
| `REKI_DATASETS_ROOT` | only for a training checkpoint: the root of the [benchmark datasets](data.md#benchmark-datasets) |
| `HF_DATASETS_CACHE` | only for a training checkpoint: as in [training](training.md#slurm) |

### `robotwin`, `robotwin_clean2rand`

RoboTwin runs in one of two setups. In both, install the simulator
([SETUP.md](../scripts/eval/SETUP.md#robotwin-20)) on the machine that runs it.

**1. The GPU nodes run the simulator and the evaluation.** Submit as for `libero`: each job
starts its servers on its own node, from `$REKI_SIM_ROOT/robotwin`.

**2. Another machine runs the simulator.** Use this setup when the GPU nodes that run the
evaluation have no Vulkan support (installing it requires administrator rights), and a separate
machine with Vulkan support can run the simulation. The evaluation connects to that machine
through the network.

- Start the servers on the sim machine
  ([SETUP.md](../scripts/eval/SETUP.md#run-the-server-on-another-machine)). They listen only on
  loopback, so reach them through an SSH tunnel, and pass the tunnel ports to `--sim_port`.
- `--sim_port 29747-29750` (or `29747,29748`) gives one worker per port, and one job per
  `--num_workers` ports. The servers use the
  task config of one difficulty, so pass `--task robotwin_easy` or `robotwin_hard`. Evaluate
  one checkpoint at a time, because a server serves one client at a time.
- With Slurm, the jobs run on compute nodes, which cannot reach the loopback of the submit
  node. Pass `--sim_relay=True` to make each tunnel port reachable on the cluster address of
  the submit node:
  it starts [`scripts/eval/sim_relay.py`](../scripts/eval/sim_relay.py) for each port of
  `--sim_port`, at the same port number. The relays keep
  running after `submit`.

  ```bash
  pixi run python launchers/sft.py select 1 eval --task robotwin_easy --num_workers 2 \
      --sim_port 29747-29750 --sim_relay=True submit
  ```

  Every node of the cluster network can then reach the servers. This exposes an unauthenticated
  pickle port on `<cluster address>:<port>`, which permits remote code execution. Relay logs are
  written in the eval output directory. Stop the relays when the evaluation ends:
  `pkill -f scripts/eval/sim_relay.py`.
- To control the exposure yourself, make the ports reachable and pass `--sim_host <address>`.
  `submit` then starts no relay. Bind a tunnel that you open from the submit node to the
  cluster address (`ssh -N -L <address>:29747:127.0.0.1:29747 <sim machine>`), or run
  `python3 scripts/eval/sim_relay.py <port> <port>` on the submit node, one per port.

#### Seed cache

At each reset, the RoboTwin server searches for a seed that the scripted expert solves, which can
take up to one hour. The seed cache stores each accepted seed and its instruction. Later
evaluations reset on it in seconds, with the same scene and instruction.

The cache is on by default, in `~/.cache/reki/robotwin_seed_cache`. To use another directory, set
`REKI_ROBOTWIN_SEED_CACHE=<dir>`. To turn it off, set `REKI_ROBOTWIN_SEED_CACHE=none`.

## Sim servers

LIBERO and LIBERO-Plus run in the client process. The other benchmarks run the simulator as a
server in its own micromamba env, because they need another Python version and other package
versions. `scripts/eval/rollout.py` drives the server over TCP
on `127.0.0.1`. A server serves one
client at a time, so each worker starts its own server.

**Never bind a server to `0.0.0.0`.** The protocol is unauthenticated pickle, so an open port lets
anyone run code on the machine. Reach a remote server through an SSH tunnel.

## Parity rules

The evaluation code follows these rules.

- **LIBERO**
  - Keep `mujoco==3.3.2`. Other versions render other colors.
  - Rotate both camera images by 180°: the env renders them upside down relative to the
    dataset.
  - The state is `[eef xyz, euler xyz, 0 pad, gripper_qpos[0]]`, as in training. The client
    converts the quaternion of the env to euler xyz and shifts the roll by π.
  - Actions are 7-dim deltas through the `robosuite` `OSC_POSE` controller (±1 = 5 cm,
    ±0.5 rad). The predicted euler rotation is converted back to axis-angle.
- **RoboTwin**
  - Set the cameras to `Large_D435`, not the upstream default `D435`.
  - Apply the three `scripts/eval/robotwin_*_fix.patch` files; `install_robotwin.sh` applies
    them.
    - Clutter fix: without it, a server that ran `dump_bin_bigbin` or `put_bottles_dustbin`
      places no clutter in later episodes, which inflates the Hard and Clutter scores.
    - Model-id fix: without it, five tasks pick objects in file-system order, so the same seed
      can give another object on another machine.
    - Arm-tag fix: the seed cache needs it for `open_laptop`, `place_object_scale` and
      `put_object_cabinet`.
  - Use the unseen instructions, as the leaderboard does.
  - The leaderboard uses 100 episodes per task; this protocol uses 20.
- **RoboCasa365**
  - The 50 target tasks (18 `atomic_seen`, 16 `composite_seen`, 16 `composite_unseen`) in the
    pretraining kitchens (`split="pretrain"`), 50 episodes per task.
  - Build the env action from the 12-dim policy order
    `[base 4, mode 1, eeΔpos 3, eeΔrot 3, gripper 1]`. Do not use `robocasa`'s `convert_action`,
    which has another order.
  - The normalization uses the statistics of the whole collection.

## LeRobot plugin

We also support inference through LeRobot's policy API:
[`lerobot_policy_reki`](../lerobot_plugin/README.md) adds the policy type `reki` to LeRobot.
`pixi install` installs it in every environment.

In the example below, `model` is the Hugging Face id of a finetuned model, e.g.
`kyutai/reki-2b-libero`, or the directory of your checkpoint after
`scripts/export_checkpoint.py --dataset <your dataset>` ([training.md](training.md#export)).
`repo_id` and `root` name its training dataset, e.g. `repo_id="HuggingFaceVLA/libero"` and
`root=$REKI_DATASETS_ROOT/HuggingFaceVLA/libero` for `kyutai/reki-2b-libero`.

**Observations.** The policy takes the features of its training dataset:

- `observation.images.<camera>`: the dataset's camera keys, at the dataset's image size.
- `observation.state`: the full dataset state. `RekiPolicy` slices and converts it as training
  does.
- `task`: the instruction.

It returns `action` in the dataset's action convention, with the dimensions the model was trained
on (`action_dim` in `reki_inference.json`). This example gives the policy one sample of its
training dataset:

```python
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.import_utils import register_third_party_plugins

register_third_party_plugins()
cfg = PreTrainedConfig.from_pretrained(model)
cfg.pretrained_path = model
dataset = LeRobotDataset(repo_id, root=root)
policy = make_policy(cfg, ds_meta=dataset.meta)
pre, post = make_pre_post_processors(cfg, pretrained_path=model)

policy.reset()  # at each episode start
frame = dataset[0]
batch = {k: v[None] for k, v in frame.items() if k.startswith("observation.")}
batch["task"] = [frame["task"]]
action = post(policy.select_action(pre(batch)))  # [1, action_dim]
```

### `lerobot-eval` on the benchmarks

To run all our benchmarks with `lerobot-eval`, use the env type `reki` of the plugin
([`env_reki.py`](../lerobot_plugin/env_reki.py)). It uses the evaluation protocol of
`scripts/eval/rollout.py`. LeRobot itself has a simulator only for LIBERO.

| Option | Meaning |
|---|---|
| `--env.benchmark` | `libero` (also LIBERO-Plus), `robotwin` or `robocasa365` |
| `--env.task` | LIBERO suites or sim tasks, comma-separated (default: all the tasks of the benchmark) |
| `--env.task_ids` | LIBERO task ids, e.g. `'[0,1]'` (default: 0-9) |
| `--env.perturbation_axis` | LIBERO-Plus: the tasks of one category, e.g. `"Camera Viewpoints"` |
| `--env.n_episodes` | episodes per task (default: the [protocol](#protocol)) |
| `--env.sim_host`, `--env.sim_port` | the sim server (RoboTwin, RoboCasa) |

- Pass `--eval.batch_size=1`. We do not currently support batched inference.
- Pass `--eval.n_episodes` = tasks × episodes per task. The env prints this number at start.
- `lerobot-eval` gives one success rate for all the episodes of a command. For the rate of each
  suite, axis or task, run one command for each.
- `lerobot-eval --seed` has no effect on the episodes. Use `--env.seed`.

**LIBERO.** Run `lerobot-eval` in the `sim` environment, after
`bash scripts/eval/install.sh libero`:

```bash
S=$(pixi run printenv REKI_SIM_ROOT)
env -u DISPLAY MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBERO_CONFIG_PATH=$S/libero/config \
  pixi run -e sim lerobot-eval --policy.path=kyutai/reki-2b-libero --env.type=reki \
    --env.benchmark=libero --env.task=libero_spatial --env.task_ids='[0]' \
    --eval.batch_size=1 --eval.n_episodes=10 --output_dir=<out_dir>
```

**LIBERO-Plus.** Run `lerobot-eval` in the `sim-plus` environment, after
`bash scripts/eval/install.sh libero_plus`. This command runs
the first 8 tasks of the `Light Conditions` category of `libero_goal`:

```bash
env -u DISPLAY MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBERO_CONFIG_PATH=$S/libero_plus/config \
  LD_PRELOAD=$PWD/.pixi/envs/sim-plus/lib/libjpeg.so.8 PYTHONPATH=$S/libero_plus/LIBERO-plus \
  pixi run -e sim-plus lerobot-eval --policy.path=kyutai/reki-2b-libero --env.type=reki \
    --env.benchmark=libero --env.task=libero_goal --env.perturbation_axis="Light Conditions" \
    --eval.batch_size=1 --eval.n_episodes=8 --output_dir=<out_dir>
```

**RoboTwin.** Start a RoboTwin server with the task config of one difficulty (`demo_clean` for
Easy, `demo_randomized` for Hard), as for [`rollout.py`](#robotwin-robotwin_clean2rand). Then run
`lerobot-eval` in the `default` environment, with the host and port of the server:

```bash
pixi run lerobot-eval --policy.path=kyutai/reki-2b-robotwin-clean --env.type=reki \
  --env.benchmark=robotwin --env.task=beat_block_hammer --env.sim_host=<host> --env.sim_port=<port> \
  --eval.batch_size=1 --eval.n_episodes=5 --output_dir=<out_dir>
```

**RoboCasa365.** Start one sim server: the `--sim_server_cmd` of the commands that
`launchers/eval_protocols.py` prints, with a file path in place of `{portfile}`. The server writes
its port to that file when it is ready. A server serves one client, then stops. Then run
`lerobot-eval` in the `default` environment:

```bash
pixi run lerobot-eval --policy.path=kyutai/reki-2b-robocasa365 --env.type=reki \
  --env.benchmark=robocasa365 --env.task=CloseFridge --env.sim_port=$(cat <portfile>) \
  --eval.batch_size=1 --eval.n_episodes=5 --output_dir=<out_dir>
```

`lerobot-eval` writes `eval_info.json`, not the `results.json` files of `rollout.py`.

The LeRobot async server does not accept plugin policies. For a real robot, use
[`scripts/deploy/policy_server_lerobot.py`](../README.md#real-robot-deployment).
