"""Dataset and simulator locations, each read from one environment variable at call time
(set in the repository's ``.env``, see ``.env.example``).

``scripts/load_env.sh`` sets the defaults. A variable that is unset raises ``RuntimeError``
naming it and what it must point to.

Required (when the feature that needs it is used):

``REKI_DATASETS_ROOT``
    LeRobot datasets root; ``DatasetSpec.root`` / ``collection`` in
    ``reki.data.robotics.registry`` hold the paths under it.

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
