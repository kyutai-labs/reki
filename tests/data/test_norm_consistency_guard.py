"""Guards against per-leaf quantiles on absolute-control collections.

If an absolute-joint collection carries per-leaf quantiles instead of one global qk band, the
same physical pose normalizes differently per leaf and silently corrupts the shared state MLP
and flow targets.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from lerobot.utils.constants import ACTION
from numpy.typing import ArrayLike

from reki.data.robotics.normalization import find_absolute_norm_violations, qk_identical_across

pytestmark = pytest.mark.cpu

ABS = "absolute joint"


def _leaf(qk01: list[float], qk99: list[float], col: str = ACTION):
    return {col: {"qk01": torch.tensor(qk01), "qk99": torch.tensor(qk99)}}


def test_per_leaf_varying_qk_on_absolute_flagged():
    stats = [_leaf([0.0, -1.0], [1.0, 2.0]), _leaf([0.0, -1.0], [1.0, 2.5])]
    v = find_absolute_norm_violations(["d", "d"], [ABS, ABS], stats, (ACTION,))
    assert len(v) == 1 and "vary across 2 leaves" in v[0]


def test_qk_identical_across():
    band = _leaf([0.0, -1.0], [1.0, 2.0])
    assert qk_identical_across([band, _leaf([0.0, -1.0], [1.0, 2.0])], (ACTION,))
    assert not qk_identical_across([band, {}], (ACTION,))
    assert not qk_identical_across([band, _leaf([0.0, -1.0], [1.0, 2.5])], (ACTION,))
    assert qk_identical_across([], (ACTION,))


N_ROWS = 6000


def _fake_leaf(
    tmp_path: Path,
    ep_bands: list[tuple[float, float]],
    counts: list[int],
    values: ArrayLike,
    col: str = ACTION,
    episodes: bool = True,
) -> Path:
    """Synthetic leaf: per-episode quantile metadata + a data column to measure clipping on."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if episodes:
        d = tmp_path / "meta" / "episodes"
        d.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    f"stats/{col}/q01": [[float(lo)] for lo, _ in ep_bands],
                    f"stats/{col}/q99": [[float(hi)] for _, hi in ep_bands],
                    f"stats/{col}/count": [[float(c)] for c in counts],
                }
            ),
            d / "episodes.parquet",
        )
    dd = tmp_path / "data" / "chunk-000"
    dd.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({col: [[float(v)] for v in np.asarray(values, dtype=float)]}),
        dd / "file-000.parquet",
    )
    return tmp_path


_SPLIT_EPS = [(-10.0, 0.0), (30.0, 40.0)]
_SPLIT_CNT = [100, 100]


def _split_values() -> np.ndarray:
    return np.concatenate([np.linspace(-10, 0, N_ROWS // 2), np.linspace(30, 40, N_ROWS // 2)])


def test_averaged_band_that_clips_is_flagged(tmp_path: Path):
    from reki.data.robotics.normalization import band_clipping_violation

    root = _fake_leaf(tmp_path, _SPLIT_EPS, _SPLIT_CNT, _split_values())
    stats = {ACTION: {"qk01": torch.tensor([10.0]), "qk99": torch.tensor([20.0])}}
    v = band_clipping_violation("d", root, stats, (ACTION,))
    assert len(v) == 1 and "per-episode AVERAGE" in v[0] and "100.0%" in v[0]
