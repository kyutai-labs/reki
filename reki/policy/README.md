# Policy

The REKI model, its configuration, and the training config, optimizer and checkpoint loading
around it. The VLM and the action expert are in [`reki/vlm/qwen3_5`](../vlm/qwen3_5/README.md);
[docs/architecture.md](../../docs/architecture.md) describes the model.

| File | Contents |
|---|---|
| `modeling.py` | `RekiModel`: batch → loss, batch → action chunk (`predict_action_chunk`), the per-episode register cache, `compile_inference`, `save_pretrained`. `RekiFlowModel`: the VLM with expert and the action, time and state projections; `forward`, `sample_actions` |
| `configuration.py` | `RekiConfig`, the model hyperparameters; `from_train_config` builds it for training, inference and evaluation |
| `flow_utils.py` | flow-matching helpers from LeRobot ([NOTICE](../../NOTICE)) |
| `constants.py` | batch keys shared with the data pipeline |
| `train_config.py` | `TrainConfig`, every training option (yaml + CLI flags), saved as `train_config.yaml` |
| `optim.py` | `EMAModel`, the optimizer parameter groups, `StochasticRoundingAdamW` |
| `checkpoint.py` | resolve a local directory or Hub id, load the weights with the EMA overlay, seed the mirrored state-MLP banks from their base banks |
| `utils.py` | `to_device` |
| `external/` | the adapter contract for training another policy ([README](../../README.md#train-your-own-model)) |

`modeling.py`, `configuration.py`, `flow_utils.py` and `constants.py` define the model. `optim.py`
is training only. Inference also uses `train_config.py` and `checkpoint.py`: a checkpoint loads
from its `train_config.yaml` (`TrainConfig.from_checkpoint`), not from `config.json`.

## Wrappers

Each layer holds the next one as an attribute; none subclasses another.

- `RekiFlowModel.vlm_with_expert` is the `Qwen3_5VLMWithExpert`.
- `RekiModel.model` is the `RekiFlowModel`.
- `reki.RekiPolicy.policy` is the `RekiModel`, next to the preprocessor, the inference spec and the
  stats ([`reki/inference/policy.py`](../inference/policy.py)).
- The LeRobot `RekiPolicy.reki` is the `reki.RekiPolicy` ([plugin](../../lerobot_plugin/README.md)).

## Entry points

- `scripts/train.py` (`pixi run train`) builds `RekiModel` from the `TrainConfig`, loads
  `pretrained_path`, trains with `optim.py` and saves `checkpoint-N/`
  ([docs/training.md](../../docs/training.md#run)).
- `scripts/dry_run.py` runs a few batches through the same builders and prints diagnostics.
