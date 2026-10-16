# Simulator setup for the evals

[`install.sh`](install.sh) installs the simulator of each benchmark that `launchers/eval_protocols.py` drives. Run it from the repository root:
```
bash scripts/eval/install.sh <benchmark>...   # libero libero_plus robotwin robocasa365
```
- Each benchmark goes to `$REKI_SIM_ROOT/<benchmark>`. The default root is `storage/sim` in the repository. To use another disk, set `REKI_SIM_ROOT` in `.env` (see [`.env.example`](../../.env.example)).
- The pins match the released numbers ([pin table](#pins-at-a-glance)).
- The script installs a benchmark once, then runs its check on each call. To install a benchmark again, for example after a failed install, delete its directory.
- To use an install that you already have, link it: `ln -s <your install> $REKI_SIM_ROOT/<benchmark>`. It must have the layout that `install.sh` makes. Do not move an install: the micromamba envs and the editable pip installs hold absolute paths.

## Overview

| task | simulator runs in | environment | render | setup |
|---|---|---|---|---|
| `libero` | the eval process | pixi `sim` | CPU (OSMesa) | [LIBERO](#libero) |
| `libero_plus` | the eval process | pixi `sim-plus` | CPU (OSMesa) | [LIBERO-Plus](#libero-plus) |
| `robotwin`, `robotwin_clean2rand` | a separate server, `scripts/eval/robotwin_sim_server.py` | pixi `default` + micromamba `robotwin` | GPU (Vulkan + CUDA) | [RoboTwin 2.0](#robotwin-20) |
| `robocasa365` | a separate server, `scripts/eval/robocasa365_sim_server.py` | pixi `default` + micromamba `robocasa365` | CPU (OSMesa) | [RoboCasa365](#robocasa365) |

## Requirements

- The pixi `default` env (`pixi install --frozen`). `install.sh` installs the `sim` and `sim-plus` envs when a benchmark needs them.
- `git`, `curl`, `unzip`, and network access to GitHub, PyPI, conda-forge and the Hugging Face Hub.
- Only to evaluate a training checkpoint: the datasets ([docs/data.md](../../docs/data.md#benchmark-datasets)). The eval reads the normalization stats and the fps from them.
- Optional: `HF_HOME` (cache for the `Qwen/Qwen3.5-2B` tokenizer/processor). The eval forwards it to the client if set.

The server benchmarks (RoboTwin, RoboCasa365) run in micromamba envs. `install.sh` puts micromamba 2.1.0 in `$REKI_SIM_ROOT/bin` and each env in `$REKI_SIM_ROOT/<benchmark>/mamba/envs/<benchmark>`. The commands below use `S` (the root) and `mrun` (run a command in the env of a benchmark):
```
S=$(pixi run printenv REKI_SIM_ROOT)
mrun() { local b=$1; shift; env PYTHONNOUSERSITE=1 PYTHONPATH= "$S/bin/micromamba" run -r "$S/$b/mamba" -n "$b" "$@"; }
```
`PYTHONNOUSERSITE=1 PYTHONPATH=` keep `~/.local` and `PYTHONPATH` from shadowing the env.

### Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `NamespaceNotFound` for `robocasa` (RoboCasa365 server) | a `robocasa/` directory in the working directory shadows the package | run the server from another directory |
| `PermissionError: '/tmp/robosuite.log'` (LIBERO, LIBERO-Plus) | robosuite 1.4 logs to `/tmp/robosuite.log` unless `macros_private.py` exists, and another user owns that file | run `bash scripts/eval/install.sh libero` (or `libero_plus`) once in this env |

## LIBERO

- Pixi env `sim`: PyPI `libero==0.1.1` (in `pixi.lock`), `robosuite` 1.4, `mujoco==3.3.2`, conda-forge `mesalib==25.0.5`. Later `mesalib` builds have no `libOSMesa.so`.
- No GPU and no apt packages. OSMesa renders on the CPU.
- **Keep MuJoCo at 3.3.2.** Other versions render different colors and lower the success rate.
- [`libero_config.sh`](libero_config.sh) writes the LIBERO and LIBERO-Plus configs and links the `assets/` of the `libero` package to `libero/assets`. Without the link, `libero` downloads the assets from the HF hub.

## LIBERO-Plus

- Pixi env `sim-plus`. It pins the runtime deps: `robosuite==1.4.0`, `bddl==1.0.1`, `robomimic==0.2.0`, `mujoco==3.3.2`, `wand` + conda `imagemagick`, `mesalib==25.0.5`.
- The eval puts the LIBERO-Plus clone on `PYTHONPATH`. **Do not pip-install it.**
- LIBERO-Plus reads its assets only from `libero/libero/assets` in the clone. `install.sh` unzips them there.

## RoboTwin 2.0

### Requirements

- An NVIDIA GPU with **Vulkan** (SAPIEN renderer) and **CUDA** (CuRobo motion planning at reset). There is no CPU path.
- VRAM: 8-13 GB per server (it grows over hours of episodes), plus about 7 GB for the policy.
- The NVIDIA graphics libraries (Vulkan), not only the CUDA ones. Install step 1 needs root.
- No system CUDA toolkit. The install script puts CUDA 12.1 and gcc 12 into the env.

### Install

1. As root, install the NVIDIA graphics libraries. They must match the kernel module version exactly. A compute-only install (`libnvidia-compute-<ver>` without `libnvidia-gl-<ver>`) does not have them.
   - Required: `libGLX_nvidia`, `libEGL_nvidia`, `libnvidia-glcore/-glsi/-tls/-glvkspirv/-eglcore`, `libnvidia-gpucomp`, `libnvidia-rtcore`, `libnvoptix`, and the manifests `/usr/share/vulkan/icd.d/nvidia_icd.json` and `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`.
   - On Ubuntu, `libnvidia-gl-<ver>` (`libnvidia-gl-<ver>-server` on the server driver branch) provides all of them.
   - Also install `libvulkan1 libgl1 libglvnd0`, `vulkan-tools` (for `vulkaninfo`), `git` and `unzip`.
   - Tested with driver 580.126.20 (`libnvidia-gl-580-server`) on H100.
2. Check that Vulkan sees the GPU. Expect the GPU name, not only `llvmpipe`.
   ```
   vulkaninfo --summary | grep deviceName
   ```
3. Run the install on a machine with the GPU. It needs no root.
   ```
   bash scripts/eval/install.sh robotwin
   ```

It runs [`install_robotwin.sh`](install_robotwin.sh), which does these steps:

- It creates the micromamba env `robotwin` in `<base>/mamba`: Python 3.10, CUDA 12.1.1, gcc 12, `torch` 2.4.1 (cu121), the RoboTwin requirements, `setuptools` 69.5.1.
- It clones RoboTwin to `<base>/RoboTwin` and applies three upstream fixes (`scripts/eval/robotwin_*_fix.patch`).
- It runs the upstream installer `script/_install.sh` with `pytorch3d` pinned. The installer builds `pytorch3d` and CuRobo v0.7.8 and patches `sapien` and `mplib`.
- It sets CuRobo `use_cuda_kernel: False`. The fused L-BFGS kernel of v0.7.8 fails on Hopper (H100) with `CUDA error 715` at MotionGen warmup.
- It downloads the assets at the pinned revision and checks their SHA-256.
- It sets the cameras of `demo_clean.yml` and `demo_randomized.yml` to `Large_D435` (640×480), the resolution used to train and evaluate our models (downscaled to 448² pixels by `image_size` in our configs). The upstream default is `D435` (320×240).


### Check

Set the server launcher `SRV` (`S` as in [Requirements](#requirements)). Then start one server (stop it with Ctrl-C), and run one episode.
```
SRV="env PYTHONNOUSERSITE=1 PYTHONPATH= NVIDIA_DRIVER_CAPABILITIES=all CUDA_HOME=$S/robotwin/mamba/envs/robotwin XDG_CACHE_HOME=/tmp/mamba_cache.$$ $S/bin/micromamba run -r $S/robotwin/mamba -n robotwin"
$SRV python -s scripts/eval/robotwin_sim_server.py --repo "$S/robotwin/RoboTwin" --task-config demo_clean \
    --port 0 --port-file /tmp/robotwin.port      # prints "Render Well", then "listening on 127.0.0.1:<port>"; Ctrl-C
pixi run python launchers/eval_protocols.py robotwin <ckpt_dir> $HOME/rt_smoke \
    --difficulties=easy --tasks=adjust_bottle --n_episodes=1 | bash
```

### Run the server on another machine

- A server holds one task config and serves one client at a time. A second client waits until the first one closes.
- Start each server from its own shell, so each gets its own `XDG_CACHE_HOME`.
- Use ports below 32768. Ports in the Linux ephemeral range (32768-60999) can fail with `EADDRINUSE`.
- **Never use `--host 0.0.0.0`.** The protocol is unauthenticated pickle, so an open port lets anyone run code on the machine.

1. On the sim machine, check out this repo, run `bash scripts/eval/install.sh robotwin`, and set `S` and `SRV` as in the check above. Start one server per port, bound to loopback.
   ```
   $SRV python -s scripts/eval/robotwin_sim_server.py --repo "$S/robotwin/RoboTwin" \
       --task-config demo_clean --host 127.0.0.1 --port 29747
   ```
2. On the eval machine (no RoboTwin install needed), open a tunnel and run the eval. A list or range of ports (`--sim_port 29747-29750`) runs one eval worker per server.
   ```
   ssh -N -L 29747:127.0.0.1:29747 <sim-machine> &
   pixi run python launchers/eval_protocols.py robotwin <ckpt_dir> <out_dir> --difficulties=easy \
       --sim_port 29747 | bash
   ```
3. With Slurm, the compute nodes cannot reach the loopback of the submit node. Opt in to a relay, or make the ports reachable yourself, as described in [the evaluation docs](../../docs/evaluation.md#robotwin-robotwin_clean2rand).
   ```
   ssh -N -L 29747:127.0.0.1:29747 <sim-machine> &
   pixi run python launchers/sft.py select 1 eval --task robotwin_easy --sim_port 29747 --sim_relay=True submit
   ```

### Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `vulkaninfo` lists only `llvmpipe` | EGL vendor lib or its manifest missing | install step 1 |
| SAPIEN `Scene()` fails with `ErrorInitializationFailed` | `libnvidia-rtcore` or `libnvoptix` missing | install step 1 |
| server imports, `reset()` fails with no supported GPU / `VkPhysicalDevice` | no ICD manifest, no `libGLX_nvidia`, or `NVIDIA_DRIVER_CAPABILITIES` without `graphics` (containers) | set `NVIDIA_DRIVER_CAPABILITIES` to `all`, check `vulkaninfo --summary` |
| first start takes minutes before `listening on` | render and CuRobo init | wait |
| server hangs at import | env or micromamba root on sshfs/NFS | keep them on local disk |
| `micromamba run` fails with `ENOLCK` | micromamba lock file on NFS | set `XDG_CACHE_HOME` to a local dir, one per server process (`SRV`) |
| log shows `[svulkan2] [error] OIDN Error: out of memory`, then `OIDN Error: changes to the filter are not committed` | GPU memory exhausted (RoboTwin enables the OIDN denoiser) | run fewer servers per GPU |
| `reset` fails with `RuntimeError: cannot create buffer` | GPU memory exhausted | run fewer servers per GPU |

## RoboCasa365

### Troubleshooting

| symptom | cause | fix |
|---|---|---|
| env build segfaults (exit 139) | `numba`'s LLVM conflicts with the preloaded Mesa `llvmpipe` | `NUMBA_DISABLE_JIT=1` |

## Pins at a glance

The pins are variables at the top of `scripts/eval/install.sh` and `scripts/eval/install_robotwin.sh`.

| component | pin |
|---|---|
| micromamba | 2.1.0 |
| LIBERO sim | PyPI `libero==0.1.1`, `mujoco==3.3.2`, `mesalib==25.0.5` (`pixi.lock`, env `sim`) |
| LIBERO assets | HF `lerobot/libero-assets` @ `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6` |
| LIBERO bddl/init | bundled in PyPI `libero==0.1.1` |
| LIBERO-Plus | `sylvestf/LIBERO-plus` @ `4976dc30028e805ff8094b55501d532c48fec182`; deps in `pixi.lock` (env `sim-plus`) |
| LIBERO-Plus assets | HF `Sylvest/LIBERO-plus` `assets.zip`, not pinned |
| RoboTwin | `RoboTwin-Platform/RoboTwin` @ `c3ddfa8b97d5519efa828b075999bd0006778e5e`; CuRobo `v0.7.8`; `torch` 2.4.1; `sapien` 3.0.0b1; CUDA 12.1.1 |
| RoboTwin `pytorch3d` | `facebookresearch/pytorch3d` @ `75ebeeaea0908c5527e7b1e305fbc7681382db47` (0.7.8) |
| RoboTwin assets | HF `TianxingChen/RoboTwin2.0` @ `9dc9299c163db059931898a9f0852098a61155a1`; SHA-256 in `install_robotwin.sh` |
| RoboTwin patches | upstream fixes `3ada624` (`robotwin_clutter_fix.patch`), `e154702` (`robotwin_arm_tag_fix.patch`) and `4bf7b48` (`robotwin_model_ids_fix.patch`), `envs/` only; CuRobo `use_cuda_kernel: False`; `Large_D435` cameras; all applied by `install_robotwin.sh` |
| RoboCasa365 | `robocasa/robocasa` @ `b4684e6ee37d377cc392e98302a6b916d588b415`; `ARISE-Initiative/robosuite` @ `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` |
| RoboCasa365 assets | `download_kitchen_assets`, not versioned upstream |
