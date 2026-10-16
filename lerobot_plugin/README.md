# LeRobot plugin

`lerobot_policy_reki` is a separate distribution ([`pyproject.toml`](pyproject.toml)) that runs a
finetuned REKI checkpoint through LeRobot's policy API, for example with `lerobot-eval`. It wraps
`reki.RekiPolicy` and is inference only. `pixi install` installs it, editable, in every environment.

## What LeRobot registers

`register_third_party_plugins()` (which `lerobot-eval` calls) imports every installed
distribution whose name starts with `lerobot_policy_`. The import registers the policy type `reki`
(`RekiConfig`) and the env type `reki` (`RekiEnv`, `--env.type=reki`).

| File | Contents |
|---|---|
| `configuration_reki.py` | `RekiConfig`: the features of the training dataset, the image key → camera map, `fps` and `n_action_steps` |
| `modeling_reki.py` | `RekiPolicy`: holds a `reki.RekiPolicy`; `select_action` predicts a chunk every `n_action_steps` steps and queues it |
| `processor_reki.py` | the pre- and postprocessor |
| `env_reki.py` | `RekiEnv`: the benchmarks of `scripts/eval/rollout.py` under `lerobot-eval`, with the same tasks, episode seeds, step cap and sim servers |
| `export.py` | writes the LeRobot files into an exported checkpoint |

## Export

With `--dataset <name>` or `--benchmark <name>`, `scripts/export_checkpoint.py`
([docs/training.md](../docs/training.md#export)) writes `reki_inference.json`,
`reki_stats.safetensors`, `policy_preprocessor.json` and `policy_postprocessor.json`, and
**replaces `config.json` with the plugin's `RekiConfig`**. The reki loader builds the model config
from `train_config.yaml`, so the export still loads in reki. The export reads the training
dataset's `meta/info.json` under `REKI_DATASETS_ROOT`. To write only the LeRobot files into an
export that has the spec and stats: `pixi run python -m lerobot_policy_reki.export <export_dir>`.

## Limits

- Batch size 1: pass `--eval.batch_size=1`.
- The images must have the image size of the training dataset.

[docs/evaluation.md](../docs/evaluation.md#lerobot-plugin) gives the Python example and the
`lerobot-eval` commands for each benchmark.
