#!/usr/bin/env bash
# Install the simulators of the released evals under $REKI_SIM_ROOT/<benchmark> (default
# <repo>/storage/sim; set it in .env). Pins of the released numbers. The script stops at the
# first error. It installs a benchmark once, then runs its check on each call. To install a
# benchmark again, delete its directory.
#
# Usage: bash scripts/eval/install.sh <benchmark>...
#   benchmarks: libero libero_plus robotwin robocasa365

set -euo pipefail

MICROMAMBA_VERSION=2.1.0-0
LIBERO_ASSETS_REVISION=0b3ea86be5fe169d0fd036ae63d1070ec09e90f6   # HF dataset lerobot/libero-assets
LIBERO_PLUS_COMMIT=4976dc30028e805ff8094b55501d532c48fec182        # sylvestf/LIBERO-plus
ROBOCASA365_ROBOSUITE_COMMIT=5ce6643f3092639d08f7b0f90ed1c6a84f50552c
ROBOCASA365_COMMIT=b4684e6ee37d377cc392e98302a6b916d588b415        # robocasa 1.0.1

BENCHMARKS="libero libero_plus robotwin robocasa365"
[ $# -ge 1 ] || { echo "usage: $0 <benchmark>... ($BENCHMARKS)" >&2; exit 2; }
for b in "$@"; do
    case " $BENCHMARKS " in *" $b "*) ;; *) echo "unknown benchmark $b ($BENCHMARKS)" >&2; exit 2 ;; esac
done

REKI="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PIXI_PROJECT_ROOT="$REKI" . "$REKI/scripts/load_env.sh"   # .env, then the storage/ defaults
ROOT="$(mkdir -p "$REKI_SIM_ROOT" && cd "$REKI_SIM_ROOT" && pwd)"
MAMBA="$ROOT/bin/micromamba"

log() { echo "[install $B] $*"; }
die() { echo "[install $B] ERROR: $*" >&2; exit 1; }
pixi_run() { pixi run --manifest-path "$REKI/pyproject.toml" --frozen "$@"; }
pixi_install() { pixi install --manifest-path "$REKI/pyproject.toml" --frozen -e "$1"; }
# micromamba lock files in the install dir, not ~/.cache (ENOLCK on NFS homes; SETUP.md).
mamba() { XDG_CACHE_HOME="$DIR/.cache" "$MAMBA" "$@"; }
# Run in the micromamba env; ~/.local and PYTHONPATH cannot shadow it.
mrun() { PYTHONNOUSERSITE=1 PYTHONPATH= mamba run -r "$DIR/mamba" -n "$B" "$@"; }
mpip() { mrun python -s -m pip "$@"; }

get_micromamba() {
    [ -x "$MAMBA" ] && return
    local arch
    case "$(uname -m)" in x86_64) arch=linux-64 ;; aarch64) arch=linux-aarch64 ;; *) die "no micromamba for $(uname -m)" ;; esac
    log "micromamba $MICROMAMBA_VERSION"
    mkdir -p "$ROOT/bin"
    curl -fsSL -o "$MAMBA.part" \
        "https://github.com/mamba-org/micromamba-releases/releases/download/$MICROMAMBA_VERSION/micromamba-$arch"
    chmod +x "$MAMBA.part"
    mv "$MAMBA.part" "$MAMBA"
}

# Render one LIBERO-10 scene on the CPU in pixi env $1; $2...: extra env assignments.
render_check() {
    (unset DISPLAY; env MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBERO_CONFIG_PATH="$DIR/config" "${@:2}" \
        pixi run --manifest-path "$REKI/pyproject.toml" --frozen -e "$1" python -c "import os; from libero.libero import benchmark, get_libero_path; from libero.libero.envs import OffScreenRenderEnv; t = benchmark.get_benchmark_dict()['libero_10']().get_task(0); e = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path('bddl_files'), t.problem_folder, t.bddl_file), camera_heights=256, camera_widths=256); e.seed(0); assert e.reset()['agentview_image'].shape == (256, 256, 3)") \
        || die "the check failed: a LIBERO-10 scene does not render"
    log "check OK: a LIBERO-10 scene renders"
}

install_libero() {
    pixi_install sim
    log "assets: lerobot/libero-assets @ $LIBERO_ASSETS_REVISION"
    pixi_run hf download lerobot/libero-assets --repo-type dataset \
        --revision "$LIBERO_ASSETS_REVISION" --local-dir "$DIR/assets"
    bash "$REKI/scripts/eval/libero_config.sh" libero "$DIR"
}
check_libero() {
    bash "$REKI/scripts/eval/libero_config.sh" libero "$DIR"
    render_check sim
}

install_libero_plus() {
    pixi_install sim-plus
    local repo="$DIR/LIBERO-plus"
    local bench="$repo/libero/libero"
    log "LIBERO-plus @ $LIBERO_PLUS_COMMIT"
    git clone -q https://github.com/sylvestf/LIBERO-plus.git "$repo"
    git -C "$repo" checkout -q "$LIBERO_PLUS_COMMIT"
    # LIBERO-Plus reads its assets from libero/libero/assets in the clone.
    log "assets: Sylvest/LIBERO-plus assets.zip (not versioned upstream)"
    pixi_run hf download Sylvest/LIBERO-plus assets.zip --repo-type dataset --local-dir "$DIR/download"
    # The zip holds the assets under a long path (inspire/hdd/.../LIBERO-plus-0/assets).
    unzip -q "$DIR/download/assets.zip" -d "$DIR/download"
    local src
    src="$(find "$DIR/download" -type d -name assets -path '*LIBERO-plus*' -print -quit)"
    [ -n "$src" ] && [ -d "$src/scenes" ] || die "no assets/scenes in assets.zip"
    mv -T "$src" "$bench/assets"
    rm -rf "$DIR/download"
    bash "$REKI/scripts/eval/libero_config.sh" libero_plus "$DIR"
}
check_libero_plus() {
    bash "$REKI/scripts/eval/libero_config.sh" libero_plus "$DIR"
    render_check sim-plus PYTHONPATH="$DIR/LIBERO-plus" LD_PRELOAD="$REKI/.pixi/envs/sim-plus/lib/libjpeg.so.8"
}

install_robotwin() {
    get_micromamba
    MICROMAMBA="$MAMBA" bash "$REKI/scripts/eval/install_robotwin.sh" "$DIR"
}
check_robotwin() {
    NVIDIA_DRIVER_CAPABILITIES=all CUDA_HOME="$DIR/mamba/envs/robotwin" \
        mrun python -s -c "import sapien; sapien.Scene()" \
        || die "the check failed: SAPIEN cannot render (Vulkan, scripts/eval/SETUP.md)"
    log "check OK: SAPIEN renders"
}

# Import the sim server of a RoboCasa benchmark, as the eval starts it.
server_check() {
    pixi_install sim   # libOSMesa.so.8
    (cd "$ROOT" && MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa NUMBA_DISABLE_JIT=1 \
        LD_PRELOAD="$REKI/.pixi/envs/sim/lib/libOSMesa.so.8" \
        mrun python -s "$REKI/scripts/eval/$1" --help >/dev/null) \
        || die "the check failed: $1 does not import"
    log "check OK: $1 imports"
}

install_robocasa365() {
    get_micromamba
    mamba create -y -q -r "$DIR/mamba" -n robocasa365 -c conda-forge python=3.11
    log "robosuite @ $ROBOCASA365_ROBOSUITE_COMMIT, robocasa @ $ROBOCASA365_COMMIT"
    git clone -q https://github.com/ARISE-Initiative/robosuite.git "$DIR/robosuite"
    git -C "$DIR/robosuite" checkout -q "$ROBOCASA365_ROBOSUITE_COMMIT"
    git clone -q https://github.com/robocasa/robocasa.git "$DIR/robocasa"
    git -C "$DIR/robocasa" checkout -q "$ROBOCASA365_COMMIT"
    mpip install -e "$DIR/robosuite"
    mpip install -e "$DIR/robocasa"   # pins numpy 2.2.5, numba 0.61.2, mujoco 3.3.1, lerobot 0.3.3
    log "macros, kitchen assets (not versioned upstream)"
    # download_kitchen_assets asks "Proceed? (y/n)" before each download.
    (cd "$DIR/robocasa" && mrun python -s -m robocasa.scripts.setup_macros \
        && printf 'y\n%.0s' {1..100} | mrun python -s -m robocasa.scripts.download_kitchen_assets)
}
check_robocasa365() { server_check robocasa365_sim_server.py; }

for B in "$@"; do
    DIR="$ROOT/$B"
    if [ -e "$DIR/.installed" ]; then
        log "$DIR is installed"
    elif [ -e "$DIR" ]; then
        die "$DIR holds an unfinished install: delete it, then run the script again"
    else
        mkdir -p "$DIR"
        "install_$B"
        rm -rf "$DIR/.cache"   # pip and micromamba caches
        touch "$DIR/.installed"
        log "installed: $DIR"
    fi
    "check_$B"
done
