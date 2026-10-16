"""scripts/export_checkpoint.py on a tiny fake checkpoint: EMA merge semantics, sidecar scrub."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from reki.policy.train_config import TrainConfig
from scripts.export_checkpoint import export_checkpoint

pytestmark = pytest.mark.cpu

TIED = {
    "model.vlm_with_expert.vlm.model.language_model.embed_tokens.weight": "model.vlm_with_expert.vlm.lm_head.weight"
}


def _fake_ckpt(tmp_path: Path) -> Path:
    ckpt = tmp_path / "checkpoint-10"
    ckpt.mkdir()
    g = torch.Generator().manual_seed(0)
    raw = {
        "model.vlm_with_expert.vlm.lm_head.weight": torch.randn(4, 3, generator=g).bfloat16(),
        "model.expert.w": torch.randn(5, 2, generator=g).bfloat16(),
        "model.action_out_proj.weight": torch.randn(3, 3, generator=g),
        "model.frozen.w": torch.randn(2, 2, generator=g).bfloat16(),
    }
    save_file(raw, str(ckpt / "model.safetensors"), metadata=TIED)
    ema = {
        "model.vlm_with_expert.vlm.model.language_model.embed_tokens.weight": torch.randn(
            4, 3, generator=g
        ),
        "model.expert.w": torch.randn(5, 2, generator=g),
        "model.action_out_proj.weight": torch.randn(3, 3, generator=g),
    }
    save_file(ema, str(ckpt / "ema.safetensors"))
    cfg = TrainConfig(
        dataset=["libero", "robotwin2_clean"],
        output_dir="/home/someone/experiments/run/train",
        pretrained_path="/home/someone/experiments/cpt/checkpoint-200000",
        vlm_model_id="/home/someone/robotics/models/qwen3.5-2b",
        state_encoding="adaln",
        state_mlp_embodiments={"single_arm_ee_euler": 8},
        action_expert_num_registers=32,
    )
    cfg.save_yaml(ckpt / "train_config.yaml")
    config_json = {
        "type": "reki",
        "vlm_model_id": "Qwen/Qwen3.5-2B",
        "device": "cuda",
        "state_encoding": "adaln",
        "history_k": 3,
        "not_a_reki_field": 1,
    }
    (ckpt / "config.json").write_text(json.dumps(config_json))
    return ckpt


def test_merge_ema_export(tmp_path: Path) -> None:
    ckpt = _fake_ckpt(tmp_path)
    out = tmp_path / "out"
    manifest = export_checkpoint(
        ckpt,
        out,
        merge_ema=True,
        pretrained_path="kyutai/reki-2b",
        relativize_prefixes=["/home/someone/robotics"],
        source_rev="test",
    )

    raw = load_file(str(ckpt / "model.safetensors"))
    ema = load_file(str(ckpt / "ema.safetensors"))
    merged = load_file(str(out / "model.safetensors"))
    assert set(merged) == set(raw)
    assert not (out / "ema.safetensors").exists()
    for k, t in merged.items():
        assert t.dtype == raw[k].dtype, k
    assert torch.equal(merged["model.expert.w"], ema["model.expert.w"].bfloat16())
    assert torch.equal(merged["model.action_out_proj.weight"], ema["model.action_out_proj.weight"])
    assert torch.equal(merged["model.frozen.w"], raw["model.frozen.w"])
    assert torch.equal(
        merged["model.vlm_with_expert.vlm.lm_head.weight"],
        raw["model.vlm_with_expert.vlm.lm_head.weight"],
    )
    with safe_open(str(out / "model.safetensors"), framework="pt") as f:  # type: ignore[attr-defined]
        assert f.metadata() == TIED
    assert manifest["ema"]["ema_keys_applied"] == 2
    assert manifest["ema"]["ema_keys_skipped_no_raw_tensor"] == [
        "model.vlm_with_expert.vlm.model.language_model.embed_tokens.weight"
    ]

    text = (out / "train_config.yaml").read_text()
    assert "/home" not in text
    scrubbed = yaml.safe_load(text)
    assert scrubbed["dataset"] == ["libero", "robotwin2_clean"]
    assert scrubbed["pretrained_path"] == "kyutai/reki-2b"
    assert scrubbed["output_dir"] == ""
    assert scrubbed["vlm_model_id"] == "models/qwen3.5-2b"
    assert scrubbed["state_mlp_embodiments"] == {"single_arm_ee_euler": 8}
    assert scrubbed["action_expert_num_registers"] == 32
    cfg = TrainConfig.from_yaml(out / "train_config.yaml")
    assert cfg.pretrained_path == "kyutai/reki-2b"
    assert set(manifest["train_config"]["replaced"]) == {
        "output_dir",
        "pretrained_path",
        "vlm_model_id",
    }

    cj = json.loads((out / "config.json").read_text())
    assert "device" not in cj
    assert "not_a_reki_field" not in cj
    assert cj["type"] == "reki" and cj["state_encoding"] == "adaln" and cj["history_k"] == 3

    manifest_text = (out / "export_manifest.json").read_text()
    for leak in ("/home", str(tmp_path)):
        assert leak not in manifest_text + text + json.dumps(cj), leak
    assert manifest["source"] == "checkpoint-10"

    files = manifest["files"]
    assert set(files) == {"model.safetensors", "train_config.yaml", "config.json"}
    assert all(len(v["sha256"]) == 64 for v in files.values())


def test_no_merge_ema_export(tmp_path: Path) -> None:
    ckpt = _fake_ckpt(tmp_path)
    out = tmp_path / "out"
    manifest = export_checkpoint(
        ckpt,
        out,
        merge_ema=False,
        pretrained_path="kyutai/reki-2b",
        relativize_prefixes=["/home/someone/robotics"],
        source_rev="test",
    )
    assert manifest["ema"] == {"merged": False}
    for name in ("model.safetensors", "ema.safetensors"):
        assert (out / name).read_bytes() == (ckpt / name).read_bytes()
