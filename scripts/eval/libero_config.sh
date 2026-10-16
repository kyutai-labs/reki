#!/usr/bin/env bash
# Write the LIBERO or LIBERO-Plus config (<sim-dir>/config/config.yaml) and robosuite's private
# macros file for headless eval. Safe to rerun. Without config.yaml, libero prompts on first
# import (a batch job fails with EOFError). The PyPI libero package has no assets/ and would
# download them, so <pkg>/assets links to <sim-dir>/assets; LIBERO-Plus has its assets in its
# clone. The macros file keeps robosuite 1.4 from logging to /tmp.
#
# Usage: libero_config.sh libero|libero_plus <sim-dir> [--if-missing]
#   <sim-dir>: $REKI_SIM_ROOT/<benchmark>. --if-missing (libero): do nothing when the config,
#   the assets link and the macros file exist.

set -euo pipefail

usage() { echo "usage: $0 libero|libero_plus <sim-dir> [--if-missing]" >&2; exit 2; }
[ $# -eq 2 ] || [ $# -eq 3 ] || usage
case "$1" in libero) PIXI_ENV=sim ;; libero_plus) PIXI_ENV=sim-plus ;; *) usage ;; esac
[ $# -eq 2 ] || [ "$1 $3" = "libero --if-missing" ] || usage

BENCHMARK="$1"
SIM_DIR="$2"
CONFIG="${SIM_DIR}/config"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Follows requires-python in pyproject.toml.
SITE_PACKAGES="${REPO_ROOT}/.pixi/envs/${PIXI_ENV}/lib/python3.12/site-packages"
ROBOSUITE="${SITE_PACKAGES}/robosuite"

if [ "$BENCHMARK" = libero ]; then
    PKG="${SITE_PACKAGES}/libero/libero"
    ASSETS="${SIM_DIR}/assets"
    if [ "${3:-}" = --if-missing ] && [ -f "${CONFIG}/config.yaml" ] \
        && [ "$(readlink "${PKG}/assets")" = "${ASSETS}" ] \
        && [ -f "${ROBOSUITE}/macros_private.py" ]; then
        exit 0
    fi
    echo "[libero_config] using pixi env: ${PIXI_ENV}"
    [ -d "${PKG}/bddl_files" ] || { echo "[libero_config] libero pkg not found at ${PKG} (pixi install -e ${PIXI_ENV} first)"; exit 1; }
else
    PKG="${SIM_DIR}/LIBERO-plus/libero/libero"
    ASSETS="${PKG}/assets"
fi

[ -d "${ROBOSUITE}" ] || { echo "robosuite pkg not found at ${ROBOSUITE} (pixi install -e ${PIXI_ENV} first)" >&2; exit 1; }
if [ ! -e "${ROBOSUITE}/macros_private.py" ]; then
    printf '%s\n' '# Its presence prevents robosuite 1.4 from creating the /tmp/robosuite.log logger.' \
        'FILE_LOGGING_LEVEL = None' > "${ROBOSUITE}/macros_private.py"
    echo "[libero_config] wrote ${ROBOSUITE}/macros_private.py"
fi

mkdir -p "${CONFIG}"
if [ "$BENCHMARK" = libero ]; then
cat > "${CONFIG}/config.yaml" <<EOF
benchmark_root: ${PKG}
bddl_files: ${PKG}/bddl_files
init_states: ${PKG}/init_files
datasets: ${ASSETS}/datasets
assets: ${ASSETS}
EOF
    echo "[libero_config] wrote ${CONFIG}/config.yaml"
    if [ "$(readlink "${PKG}/assets")" != "${ASSETS}" ]; then
        ln -sfn "${ASSETS}" "${PKG}/assets"
        echo "[libero_config] symlinked ${PKG}/assets -> ${ASSETS}"
    fi
else
cat > "${CONFIG}/config.yaml" <<EOF
assets: ${ASSETS}
benchmark_root: ${PKG}
bddl_files: ${PKG}/bddl_files
datasets: ${SIM_DIR}/datasets
init_states: ${PKG}/init_files
EOF
fi
