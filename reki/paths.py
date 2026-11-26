"""Dataset and simulator locations, each read from one environment variable at call time
(set in the repository's ``.env``, see ``.env.example``).

``scripts/load_env.sh`` sets the defaults. A variable that is unset raises ``RuntimeError``
naming it and what it must point to.

Required (when the feature that needs it is used):

``REKI_DATASETS_ROOT``
    LeRobot datasets root; ``DatasetSpec.root`` / ``collection`` in
    ``reki.data.robotics.registry`` hold the paths under it.

Optional:

``FINEVISION_PATH``, ``LLAVA_EXT_PATH``, ``VL_SPATIAL_PARQUET_PATH``
    Vision-language data of pretraining (``vl_datasets``): roots holding one ``<config>/``
    dir per config of ``HuggingFaceM4/FineVision``, ``mvp-lab/LLaVA-OneVision-1.5-Instruct-Data``
    and the FineVision-format robopoint / refspatial / pixmo build. Default: ``FINEVISION_DIR``,
    ``LLAVA_EXT_DIR`` and ``VL_SPATIAL_DIR`` under ``REKI_DATASETS_ROOT``, where
    ``scripts/data/pretrain_data.py`` writes them. Layouts read by
    ``reki.data.utils.load_vl_config``.

``REKI_SIM_ROOT``
    Simulator installs, one directory per benchmark (``scripts/eval/install.sh``,
    ``scripts/eval/SETUP.md``). Read by ``launchers/eval_protocols.py``.

Other ``REKI_*`` variables, including ``REKI_RUNS_DIR``, ``REKI_ROBOTWIN_SEED_CACHE``, and
launcher resource settings, are listed in ``.env.example``. Read by HF itself, not here:
``HF_HOME`` (model/tokenizer cache), ``HF_DATASETS_CACHE``.
"""

from __future__ import annotations

import os
from pathlib import Path


def _require(var: str, what: str) -> str:
    value = os.environ.get(var)
    if not value:
        raise RuntimeError(f"${var} is unset; set it in .env to {what}")
    return value


def datasets_root() -> Path:
    """The configured LeRobot datasets root."""
    return Path(_require("REKI_DATASETS_ROOT", "the LeRobot datasets root"))


def sim_root() -> Path:
    """The configured simulator-installations root."""
    return Path(_require("REKI_SIM_ROOT", "the simulator installs root"))


FINEVISION_DIR = "vl/finevision"
LLAVA_EXT_DIR = "vl/llava_onevision"
VL_SPATIAL_DIR = "vl/spatial"
"""The VL roots under the datasets root."""


def finevision_path() -> str:
    return os.environ.get("FINEVISION_PATH") or str(datasets_root() / FINEVISION_DIR)


def llava_ext_path() -> str:
    return os.environ.get("LLAVA_EXT_PATH") or str(datasets_root() / LLAVA_EXT_DIR)


def vl_spatial_parquet_path() -> str:
    return os.environ.get("VL_SPATIAL_PARQUET_PATH") or str(datasets_root() / VL_SPATIAL_DIR)
