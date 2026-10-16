"""Export a training checkpoint directory into the HF upload layout.

    pixi run python scripts/export_checkpoint.py <ckpt_dir> <out_dir> --merge-ema \\
        --pretrained-path kyutai/reki-2b --dataset <name>
    pixi run python scripts/export_checkpoint.py <ckpt_dir> <out_dir> --merge-ema \\
        --pretrained-path kyutai/reki-2b --benchmark libero
    pixi run python scripts/export_checkpoint.py <ckpt_dir> <out_dir> --no-merge-ema

Writes:
  model.safetensors          the weights; --merge-ema puts the EMA in place of the raw weights,
                             --no-merge-ema keeps ema.safetensors next to them
  train_config.yaml          private paths removed, ``pretrained_path`` set to --pretrained-path
  config.json                the model config (the LeRobot config with --dataset / --benchmark)
  reki_inference.json        with --dataset / --benchmark: normalization stats (needs
  + reki_stats.safetensors   $REKI_DATASETS_ROOT), cameras, prompt fields, cadence
  policy_*processor.json     with --dataset / --benchmark: the LeRobot processors
  export_manifest.json       source, code revision, per-file sha256
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml
from safetensors import safe_open
from safetensors.torch import save_file

PRIVATE_PATH_MARKERS: tuple[str, ...] = ("/home", "/lustre")
CLEARED_TRAIN_KEYS: dict[str, Any] = {
    "output_dir": "",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_rev(source_rev: str | None) -> str:
    """The code revision the export is made from: ``source_rev`` when given, else the checkout's
    HEAD. Outside a git checkout (an sdist or archive install) ``source_rev`` is required."""
    if source_rev is not None:
        return source_rev
    repo = Path(__file__).resolve().parents[1]
    if not (repo / ".git").exists():
        raise ValueError(
            f"{repo} is not a git checkout; pass --source-rev <the code revision this export "
            "is made from>"
        )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def export_weights(ckpt_dir: Path, out_dir: Path, merge_ema: bool) -> dict[str, Any]:
    """Write ``model.safetensors`` (and ``ema.safetensors`` without the merge); return the EMA
    bookkeeping for the manifest."""
    raw_path = ckpt_dir / "model.safetensors"
    ema_path = ckpt_dir / "ema.safetensors"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    if not ema_path.exists():
        raise FileNotFoundError(ema_path)
    if not merge_ema:
        shutil.copyfile(raw_path, out_dir / "model.safetensors")
        shutil.copyfile(ema_path, out_dir / "ema.safetensors")
        return {"merged": False}

    tensors: dict[str, Any] = {}
    applied: list[str] = []
    with (
        safe_open(str(raw_path), framework="pt") as raw,
        safe_open(str(ema_path), framework="pt") as ema,
    ):  # type: ignore[attr-defined]
        metadata = raw.metadata()
        ema_keys = set(ema.keys())
        for key in raw.keys():
            t = raw.get_tensor(key)
            if key in ema_keys:
                shadow = ema.get_tensor(key)
                if shadow.shape != t.shape:
                    raise ValueError(
                        f"EMA shape mismatch for {key}: ema {tuple(shadow.shape)} vs raw "
                        f"{tuple(t.shape)}"
                    )
                t = shadow.to(t.dtype)
                applied.append(key)
            tensors[key] = t.contiguous()
        skipped = sorted(ema_keys - set(tensors))
    save_file(tensors, str(out_dir / "model.safetensors"), metadata=metadata)
    return {
        "merged": True,
        "ema_keys_applied": len(applied),
        "ema_keys_total": len(ema_keys),
        "ema_keys_skipped_no_raw_tensor": skipped,
        "safetensors_metadata": metadata,
    }


def _has_private_path(v: Any) -> bool:
    if isinstance(v, str):
        return v.startswith(("/", "~")) or any(m in v for m in PRIVATE_PATH_MARKERS)
    if isinstance(v, (list, tuple)):
        return any(_has_private_path(x) for x in v)
    if isinstance(v, dict):
        return any(_has_private_path(x) for x in v.values())
    return False


def _relativize(v: Any, prefixes: list[str]) -> Any:
    if isinstance(v, str):
        for prefix in prefixes:
            if v.startswith(prefix):
                return str(Path(v).relative_to(prefix))
        return v
    if isinstance(v, list):
        return [_relativize(x, prefixes) for x in v]
    if isinstance(v, dict):
        return {k: _relativize(x, prefixes) for k, x in v.items()}
    return v


def scrub_train_config(
    ckpt_dir: Path,
    out_dir: Path,
    pretrained_path: str | None,
    relativize_prefixes: list[str],
) -> dict[str, Any]:
    """Write the scrubbed ``train_config.yaml``; return the scrub record for the manifest."""
    from reki.policy.train_config import TrainConfig

    with open(ckpt_dir / "train_config.yaml") as f:
        cfg = yaml.safe_load(f)
    replaced: dict[str, Any] = {}

    for k, v in CLEARED_TRAIN_KEYS.items():
        if cfg.get(k):
            cfg[k] = v
            replaced[k] = v
    if cfg.get("pretrained_path"):
        if pretrained_path is None:
            raise ValueError(
                "the checkpoint's train_config.yaml has a pretrained_path; pass "
                "--pretrained-path <HF id of the warm-start checkpoint> to replace it"
            )
        cfg["pretrained_path"] = pretrained_path
        replaced["pretrained_path"] = pretrained_path
    for k, v in cfg.items():
        if k in replaced or not _has_private_path(v):
            continue
        nv = _relativize(v, relativize_prefixes)
        if _has_private_path(nv):
            raise ValueError(
                f"train_config.yaml key {k!r} holds a private path that no --relativize-prefix "
                "covers; add the prefix or fix the value upstream"
            )
        cfg[k] = nv
        replaced[k] = nv

    out = out_dir / "train_config.yaml"
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    TrainConfig.from_yaml(out)
    return {"replaced": replaced}


def scrub_config_json(ckpt_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Write the scrubbed ``config.json``; return the dropped keys for the manifest."""
    import dataclasses

    from reki.policy.configuration import RekiConfig

    with open(ckpt_dir / "config.json") as f:
        raw = json.load(f)
    known = {f.name for f in dataclasses.fields(RekiConfig)}
    # device is the training host's; the loader picks its own.
    keep = {k: v for k, v in raw.items() if k in known and k != "device"}
    dropped = sorted(k for k in raw if k not in keep and k != "type")
    keep = {"type": "reki", **keep}
    if _has_private_path(keep):
        raise ValueError(f"config.json still holds a private path: {keep}")
    with open(out_dir / "config.json", "w") as f:
        json.dump(keep, f, indent=4)
        f.write("\n")
    return {"dropped_keys": len(dropped)}


def export_inference_spec(
    out_dir: Path, *, dataset: str | None = None, benchmark: str | None = None
) -> dict[str, Any]:
    """Write ``reki_inference.json`` + ``reki_stats.safetensors``; return the spec."""
    from dataclasses import asdict

    from reki.inference.spec import build_inference_spec, save_inference_spec
    from reki.policy.train_config import TrainConfig

    spec, stats = build_inference_spec(
        TrainConfig.from_yaml(out_dir / "train_config.yaml"), dataset=dataset, benchmark=benchmark
    )
    save_inference_spec(out_dir, spec, stats)
    return asdict(spec)


def export_checkpoint(
    ckpt_dir: str | Path,
    out_dir: str | Path,
    merge_ema: bool,
    pretrained_path: str | None = None,
    relativize_prefixes: list[str] | None = None,
    source_rev: str | None = None,
    benchmark: str | None = None,
    dataset: str | None = None,
) -> dict[str, Any]:
    """Write the files listed in the module docstring to ``out_dir``; returns the manifest."""
    if benchmark is not None and dataset is not None:
        raise ValueError("pass --dataset or --benchmark, not both")
    ckpt_dir = Path(ckpt_dir)
    out_dir = Path(out_dir)
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(ckpt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "source": ckpt_dir.resolve().name,
        "source_rev": _source_rev(source_rev),
        "merge_ema": merge_ema,
        "ema": export_weights(ckpt_dir, out_dir, merge_ema),
        "train_config": scrub_train_config(
            ckpt_dir, out_dir, pretrained_path, relativize_prefixes or []
        ),
        "config_json": scrub_config_json(ckpt_dir, out_dir),
    }
    if benchmark is not None or dataset is not None:
        from lerobot_policy_reki.export import write_lerobot_files

        manifest["inference"] = export_inference_spec(out_dir, dataset=dataset, benchmark=benchmark)
        manifest["lerobot"] = write_lerobot_files(out_dir)
    manifest["files"] = {
        p.name: {"sha256": _sha256(p), "bytes": p.stat().st_size}
        for p in sorted(out_dir.iterdir())
        if p.name != "export_manifest.json"
    }
    if _has_private_path(manifest):
        raise ValueError(f"export_manifest.json would hold an absolute path: {manifest}")
    with open(out_dir / "export_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("ckpt_dir")
    parser.add_argument("out_dir")
    parser.add_argument(
        "--merge-ema",
        action=argparse.BooleanOptionalAction,
        required=True,
        help=(
            "overlay the EMA shadow into model.safetensors; --no-merge-ema copies "
            "model.safetensors and ema.safetensors unchanged"
        ),
    )
    parser.add_argument(
        "--pretrained-path",
        default=None,
        help="replacement for the yaml's pretrained_path (the warm-start checkpoint's HF id)",
    )
    parser.add_argument(
        "--relativize-prefix",
        action="append",
        default=[],
        metavar="PREFIX",
        help="path values under PREFIX become relative to it (e.g. the training checkout root)",
    )
    parser.add_argument(
        "--source-rev",
        "--source_rev",
        dest="source_rev",
        default=None,
        help="code revision recorded in the manifest; required outside a git checkout",
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--dataset",
        default=None,
        help="finetuning dataset (any registered dataset the checkpoint was trained on): also "
        "write the RekiPolicy inference spec + normalization stats",
    )
    target.add_argument(
        "--benchmark",
        default=None,
        choices=["libero", "robotwin", "robocasa365"],
        help="sim benchmark preset: as --dataset, plus the sim's task names",
    )
    args = parser.parse_args()
    manifest = export_checkpoint(
        args.ckpt_dir,
        args.out_dir,
        merge_ema=args.merge_ema,
        pretrained_path=args.pretrained_path,
        relativize_prefixes=args.relativize_prefix,
        source_rev=args.source_rev,
        benchmark=args.benchmark,
        dataset=args.dataset,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
