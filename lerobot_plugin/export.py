"""Make an exported reki checkpoint loadable by LeRobot (``make_policy``, ``lerobot-eval``).

    pixi run python -m lerobot_policy_reki.export <export_dir>

Overwrites ``config.json`` with a ``RekiConfig`` and writes ``policy_preprocessor.json`` +
``policy_postprocessor.json``. The reki loader reads ``train_config.yaml`` and does not use
``config.json``. The features come from the training dataset of ``reki_inference.json``: its
image keys and sizes, its full ``observation.state`` and the action ``RekiPolicy.predict``
returns. The export reads the dataset's ``meta/info.json`` under ``$REKI_DATASETS_ROOT``.
``scripts/export_checkpoint.py --dataset/--benchmark`` runs this step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from lerobot_policy_reki.configuration_reki import RekiConfig
from lerobot_policy_reki.processor_reki import make_reki_pre_post_processors
from reki.inference.spec import (
    INFERENCE_SPEC_FILE,
    dataset_fields,
    feature_names,
    load_inference_spec,
    trained_leaves,
)
from reki.policy.train_config import TrainConfig


def _one(values: set, what: str, dataset: str):
    if len(values) != 1:
        raise ValueError(f"{dataset}: leaves differ in {what}: {sorted(values)}")
    return next(iter(values))


def lerobot_config(ckpt_dir: Path) -> RekiConfig:
    """The ``RekiConfig`` of an exported checkpoint, from its spec and its training dataset."""
    if not (ckpt_dir / INFERENCE_SPEC_FILE).exists():
        raise FileNotFoundError(
            f"{ckpt_dir} has no {INFERENCE_SPEC_FILE}, so no robot: no state layout, cameras or "
            "normalization stats. The pretrained kyutai/reki-2b is such a checkpoint: finetune "
            "it on your robot first."
        )
    spec, _ = load_inference_spec(ckpt_dir)
    train_cfg = TrainConfig.from_checkpoint(ckpt_dir)
    leaves = trained_leaves(spec.dataset, train_cfg)
    fields = dataset_fields(spec.dataset, leaves, train_cfg)
    if fields.camera_keys != spec.camera_keys:
        raise ValueError(f"{spec.dataset}: cameras {fields.camera_keys} != {spec.camera_keys}")
    infos = [json.loads((p / "meta" / "info.json").read_text()) for p in leaves]

    camera_map = {
        f"{OBS_IMAGES}.{name}": key
        for name, key in zip(fields.camera_names, fields.camera_keys, strict=True)
    }
    images = {}
    for image_key in camera_map:
        h, w, c = _one(
            {tuple(i["features"][image_key]["shape"]) for i in infos}, image_key, spec.dataset
        )
        if c != 3:
            raise ValueError(f"{spec.dataset}: {image_key} is not H×W×3: {(h, w, c)}")
        images[image_key] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, h, w))

    config = RekiConfig(
        input_features={
            **images,
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(fields.state_dim,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(spec.action_dim,))},
        camera_map=camera_map,
        state_names=feature_names(infos[0], OBS_STATE),
        action_names=fields.action_names,
        fps=spec.prompt_fps,
        n_action_steps=spec.n_action_steps,
    )
    config.device = "cuda"  # __post_init__ falls back to the export host's device
    config.validate_features()
    return config


def write_lerobot_files(ckpt_dir: Path) -> list[str]:
    """Write the LeRobot config + processors into ``ckpt_dir``; return the file names."""
    config = lerobot_config(ckpt_dir)
    config._save_pretrained(ckpt_dir)
    for pipeline in make_reki_pre_post_processors(config):
        pipeline.save_pretrained(ckpt_dir)
    return ["config.json", "policy_preprocessor.json", "policy_postprocessor.json"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_dir", type=Path)
    print(write_lerobot_files(parser.parse_args().export_dir))


if __name__ == "__main__":
    main()
