# Data scripts

Run every script from the repository root:

```sh
pixi run --frozen python scripts/data/<script>.py
```

See [the data guide](../../docs/data.md) for dataset paths, registration, and the exact
per-dataset steps in the "Benchmark datasets" table.

For released benchmark datasets, download the raw inputs (`download.py` for LIBERO and
RoboCasa365, `hf download` for RoboTwin), then run the preparation script:
`prepare_robotwin.py` or `prepare_robocasa365.py`. These convert to
v3.0 and write the `qk` quantiles. Run `recompute_euler_stats.py` for axis-angle datasets such
as LIBERO, then `check_datasets.py`. LIBERO needs no preparation step.

For your own data, follow [Onboarding a dataset](../../docs/data.md#onboarding-a-dataset):
`build_v30.py` → `augment_quantiles.py` → register → `recompute_euler_stats.py` if axis-angle →
`check_datasets.py`.

## Reference

- `download.py` downloads the raw LIBERO and RoboCasa365 inputs.
- `prepare_*.py` convert and prepare the released RoboTwin and RoboCasa datasets.
- `build_v30.py` rebuilds v2.x metadata in the repository's v3.0 layout.
- `augment_quantiles.py` writes `qk` action and state quantiles into `meta/stats.json`.
- `recompute_euler_stats.py` writes euler-space normalization sidecars for axis-angle data.
- `derive_libero_mirror_axis.py` measures LIBERO's stored left-right EE-position axis.
- `check_datasets.py` runs opt-in registry and real-data sanity checks.
- `shared.py` contains conversion, quantile, and rotation helpers used by these scripts.
- `sbatch/` contains Slurm launchers for the expensive preparation and stats jobs.
