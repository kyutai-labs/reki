#!/usr/bin/env bash
# Install the RoboTwin 2.0 sim server env into <base_dir> ($REKI_SIM_ROOT/robotwin when
# `scripts/eval/install.sh robotwin` runs it):
#   <base_dir>/mamba/envs/robotwin  micromamba env (Python 3.10, CUDA 12.1, torch, SAPIEN, CuRobo)
#   <base_dir>/RoboTwin             RoboTwin clone + assets, patched for the released evals
# Needs MICROMAMBA, network, and the NVIDIA graphics libraries (scripts/eval/SETUP.md).
# No root. Run it once into an empty <base_dir>. It stops at the first error.
#
# Usage: MICROMAMBA=<micromamba binary> bash scripts/eval/install_robotwin.sh <base_dir>

set -euo pipefail

# Pins of the released numbers.
ROBOTWIN_COMMIT=c3ddfa8b97d5519efa828b075999bd0006778e5e
PYTORCH3D_COMMIT=75ebeeaea0908c5527e7b1e305fbc7681382db47   # 0.7.8
ASSETS_REVISION=9dc9299c163db059931898a9f0852098a61155a1      # HF dataset TianxingChen/RoboTwin2.0
CUDA_CHANNEL=nvidia/label/cuda-12.1.1
TORCH="torch==2.4.1 torchvision==0.19.1"
TORCH_INDEX=https://download.pytorch.org/whl/cu121
SETUPTOOLS=setuptools==69.5.1   # sapien 3.0.0b1 imports pkg_resources, which setuptools>=81 removed
declare -A ASSET_SHA256=(
    [background_texture]=54ede0fb5b783e0faa2bc98720d3affd6ca3bb9280b225b48c1aafaf31473070
    [embodiments]=6b87d7d55e106d8ff25917e0538eb1e177fc549280e8a742a8cec3cb9f953fc6
    [objects]=6aa56b3cf1e1064f7c809308144da36b00815f8b137fef2d7e4de856f8becf27
)

[ $# -eq 1 ] || { echo "usage: MICROMAMBA=<micromamba> $0 <base_dir>" >&2; exit 2; }
: "${MICROMAMBA:?set MICROMAMBA to the micromamba binary}"
REKI="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$1"
BASE="$(cd "$1" && pwd)"
ENV="$BASE/mamba/envs/robotwin"
RT="$BASE/RoboTwin"
if [ -e "$ENV" ] || [ -e "$RT" ]; then
    echo "$BASE already holds an install; use an empty dir" >&2; exit 1
fi
export XDG_CACHE_HOME="$BASE/.cache"   # micromamba locks: not ~/.cache (ENOLCK on NFS homes)

log() { echo "[install_robotwin] $*"; }
die() { echo "[install_robotwin] ERROR: $*" >&2; exit 1; }
mrun() { env PYTHONNOUSERSITE=1 PYTHONPATH= "$MICROMAMBA" run -r "$BASE/mamba" -n robotwin "$@"; }
mpip() { mrun python -s -m pip "$@"; }

log "env: Python 3.10, CUDA 12.1.1, gcc 12"
# All CUDA packages come from one version-locked channel: bare -c nvidia pulls 13.x
# sub-packages. CUDA 12.1 nvcc rejects host gcc > 12, so the env gets its own gcc 12.
"$MICROMAMBA" create -y -r "$BASE/mamba" -n robotwin -c conda-forge python=3.10
"$MICROMAMBA" install -y -r "$BASE/mamba" -n robotwin -c "$CUDA_CHANNEL" -c conda-forge \
    cuda-toolkit gcc_linux-64=12 gxx_linux-64=12 sysroot_linux-64
# `micromamba run` does not inherit CPATH: export the CUDA paths and the host compiler on activate.
mkdir -p "$ENV/etc/conda/activate.d"
cat > "$ENV/etc/conda/activate.d/zz_cuda.sh" <<'EOF'
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:${CPATH:-}"
EOF
cat > "$ENV/etc/conda/activate.d/zzz_nvcc.sh" <<'EOF'
export NVCC_PREPEND_FLAGS="-ccbin ${CC:-$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc} ${NVCC_PREPEND_FLAGS:-}"
export CUDAHOSTCXX="${CXX:-$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++}"
EOF
mrun nvcc --version | grep -q 'release 12.1' || die "nvcc is not CUDA 12.1"
if "$MICROMAMBA" list -r "$BASE/mamba" -n robotwin | grep -E '^ *(cuda-|libcu|libnv)' | grep -v "$CUDA_CHANNEL"; then
    die "the CUDA packages above do not come from $CUDA_CHANNEL"
fi

log "RoboTwin @ $ROBOTWIN_COMMIT"
git clone https://github.com/RoboTwin-Platform/RoboTwin.git "$RT"
git -C "$RT" checkout -q "$ROBOTWIN_COMMIT"
cd "$RT"
# Upstream issue #288: get_cluttered_table() mutates its default xlim/ylim lists.
git apply "$REKI/scripts/eval/robotwin_clutter_fix.patch"
# Upstream e154702 (PR #488), envs/ part: three tasks set arm_tag in load_actors, so their
# success check works after a seed-cache reset, which skips play_once.
git apply "$REKI/scripts/eval/robotwin_arm_tag_fix.patch"
# Upstream 4bf7b48 (PR #490), envs/ part: sorted model ids, so a seed picks the same object on
# every filesystem.
git apply "$REKI/scripts/eval/robotwin_model_ids_fix.patch"
sed -i "s#pytorch3d.git@stable#pytorch3d.git@$PYTORCH3D_COMMIT#" script/_install.sh
grep -q "pytorch3d.git@$PYTORCH3D_COMMIT" script/_install.sh || die "pytorch3d pin not applied"

log "pip: torch, RoboTwin requirements"
# torch first from the cu121 index, else requirements.txt pulls another CUDA build.
mpip install $TORCH --index-url "$TORCH_INDEX"
mpip install -r script/requirements.txt
mpip install "$SETUPTOOLS"

log "upstream installer: pytorch3d, sapien/mplib source patches, CuRobo v0.7.8"
# script/_install.sh has no `set -e`: check its two builds by hand.
mrun bash script/_install.sh 2>&1 | tee "$BASE/install_upstream.log"
for pkg in pytorch3d curobo; do
    grep -qE "Successfully (built|installed).*$pkg" "$BASE/install_upstream.log" \
        || die "$pkg did not build; see $BASE/install_upstream.log"
done

# The fused L-BFGS kernel of CuRobo v0.7.8 fails on Hopper (CUDA error 715 at MotionGen
# warmup): use the PyTorch L-BFGS. One of the files spells the value `true`.
log "CuRobo: use_cuda_kernel: False"
CUROBO_TASK="$RT/envs/curobo/src/curobo/content/configs/task"
sed -i -E 's/use_cuda_kernel:[[:space:]]*[Tt]rue/use_cuda_kernel: False/' \
    "$CUROBO_TASK"/{gradient_trajopt,finetune_trajopt,finetune_trajopt_slow,gradient_ik,gradient_ik_autotune,gradient_mpc}.yml
if grep -rniE 'use_cuda_kernel:[[:space:]]*true' "$CUROBO_TASK"; then die "CuRobo still uses the CUDA kernel"; fi

log "assets: TianxingChen/RoboTwin2.0 @ $ASSETS_REVISION"
sed -i "s#repo_type=\"dataset\",#repo_type=\"dataset\", revision=\"$ASSETS_REVISION\",#" assets/_download.py
grep -q "$ASSETS_REVISION" assets/_download.py || die "asset revision pin not applied"
(
    cd assets
    mrun python -s _download.py
    for z in "${!ASSET_SHA256[@]}"; do
        echo "${ASSET_SHA256[$z]}  $z.zip" | sha256sum -c -
        unzip -q -o "$z.zip"
        rm -f "$z.zip"
    done
)
# Writes absolute paths into the embodiment configs: rerun it if <base_dir> moves.
mrun python -s script/update_embodiment_config_path.py

# The released checkpoints see 640x480 (Large_D435); upstream defaults to D435 (320x240).
log "cameras: Large_D435"
for yml in task_config/demo_clean.yml task_config/demo_randomized.yml; do
    sed -i -E 's/^(\s*(head|wrist)_camera_type:\s*)D435\s*$/\1Large_D435/' "$yml"
    [ "$(grep -cE '^\s*(head|wrist)_camera_type:\s*Large_D435\s*$' "$yml")" -eq 2 ] || die "$yml: camera edit failed"
done

log "done: $BASE"
