#!/usr/bin/env python
"""Build the pretraining data of configs/cpt.yaml (docs/pretraining_data.md).

    pixi run -e data python scripts/data/pretrain_data.py <stage> <dataset|group> [options]

Run the stages in this order:

  download   the pinned upstream files, into $REKI_DATASETS_ROOT/upstream/
  build      the LeRobot datasets (--shard i/n for part i of n)
  curate     the episodes excluded from training
  stats      the normalization statistics
  verify     the counts, the files and one training sample

stats and verify of egodex, egoverse and xperience need --config.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

from pretrain import catalog
from pretrain.common import dedup, detect, download, drops, stats, v30, verify
from pretrain.recipe import VlReader

from reki.paths import datasets_root
from reki.policy.train_config import TrainConfig

logger = logging.getLogger("pretrain_data")

STAGES = ("download", "build", "curate", "stats", "verify")


def parse_shard(spec: str) -> tuple[int, int]:
    """``"i/n"`` → ``(i, n)`` with ``0 <= i < n``."""
    i, n = (int(x) for x in spec.split("/"))
    if not 0 <= i < n:
        raise ValueError(f"shard {spec}: need 0 <= i < n")
    return i, n


def resolve(spec: str) -> list[str]:
    """The datasets of a dataset or group name (registry or VL), or of several, comma-separated;
    each must have a build config."""
    from reki.data.robotics.registry import DATASET_GROUPS

    recipes = catalog.recipes()
    groups = {**{g: list(n) for g, n in DATASET_GROUPS.items()}, **catalog.groups()}
    buildable = {g for g, n in groups.items() if all(x in recipes for x in n)}
    names: list[str] = []
    for part in spec.split(","):
        if part in buildable:
            found = groups[part]
        elif part in recipes:
            found = [part]
        else:
            raise SystemExit(
                f"{part!r}: no pretraining build. Groups: {', '.join(sorted(buildable))}; "
                "datasets: docs/pretraining_data.md#build"
            )
        names += [n for n in found if n not in names]
    return names


def leaf_drops(
    name: str, root: Path, subset: bool, leaves: list[Path], detect_again: bool = False
) -> tuple[list[list[int]], set[dedup.Key]]:
    """Per leaf, the episodes that the list of excluded episodes of ``name`` drops; and the
    dataset's keys.
    ``detect_again``: make the list again first. A ``--subset`` has no list file: its list is
    made from the detectors each time."""
    recipe, pin = catalog.recipes()[name], catalog.pin(name, subset)
    groups = drops.groups(
        leaves,
        lambda leaf: recipe.reader.dedup_labels(name, root, pin, leaf),
        recipe.reader.key_columns(),
    )
    listed = drops.load(name)
    if detect_again or subset:
        hits = {
            (i, ep)
            for i, leaf in enumerate(leaves)
            for d in recipe.curate
            for ep in detect.DETECTORS[d](leaf, name)
        }
        listed = drops.detect(groups, hits, None if subset else listed)
        if not subset:
            drops.save(name, listed)
    if listed is None:
        raise FileNotFoundError(
            f"{drops.path(name)}: no list of excluded episodes. Please open an issue at "
            "https://github.com/kyutai-labs/reki/issues"
        )
    return drops.apply(listed, groups, len(leaves)), set(groups)


def curate(name: str, root: Path, subset: bool, detect_again: bool) -> set[dedup.Key]:
    """The list of excluded episodes → each leaf's ``meta/dropped_episodes.json``, then the
    reader's own curation. Returns the dataset's keys."""
    recipe, pin = catalog.recipes()[name], catalog.pin(name, subset)
    leaves = recipe.reader.leaves(name, root, pin)
    per_leaf, keys = leaf_drops(name, root, subset, leaves, detect_again)
    for leaf, eps in zip(leaves, per_leaf):
        (leaf / "meta" / "dropped_episodes.json").write_text(json.dumps(eps))
        logger.info("%s: %d dropped", leaf.name, len(eps))
    recipe.reader.curate(name, root, pin)
    return keys


def run_vl(stage: str, name: str, root: Path, subset: bool, shard: tuple[int, int]) -> None:
    """One stage of a VL dataset: download, build, verify; the others do nothing."""
    reader, pin = catalog.recipes()[name].reader, catalog.pin(name, subset)
    assert isinstance(reader, VlReader)
    if stage == "download":
        reader.download(name, root, pin)
    elif stage == "build":
        reader.build(name, root, pin, shard)
    elif stage == "verify":
        reader.verify(name, root, pin)
    else:
        logger.info("%s: %s does nothing for VL data (not LeRobot leaves)", name, stage)


def last_shard(root: Path, name: str, shard: tuple[int, int]) -> bool:
    """Marks shard ``i/n`` of ``name`` built (``<root>/.work/shards/<name>/<n>/``). True for one
    shard only: the first to see all ``n`` built."""
    i, n = shard
    marks = root / ".work" / "shards" / name / str(n)
    marks.mkdir(parents=True, exist_ok=True)
    (marks / f"{i}.done").touch()
    if not all((marks / f"{k}.done").exists() for k in range(n)):
        return False
    try:
        os.close(os.open(marks / "finish", os.O_CREAT | os.O_EXCL))
    except FileExistsError:
        return False
    return True


def run_stage(
    stage: str,
    name: str,
    root: Path,
    subset: bool,
    shard: tuple[int, int],
    train: TrainConfig,
) -> None:
    recipe, pin = catalog.recipes()[name], catalog.pin(name, subset)
    if stage == "download":
        recipe.reader.download(name, root, pin)
    elif stage == "build":
        recipe.reader.build(name, root, pin, shard)
        if shard != (0, 1) and recipe.reader.finish_after_shards and last_shard(root, name, shard):
            logger.info("%s: all %d shards are built; writing the leaves", name, shard[1])
            marks = root / ".work" / "shards" / name / str(shard[1])
            try:
                recipe.reader.build(name, root, pin, (0, 1))
            except BaseException:
                (marks / "finish").unlink()
                raise
            shutil.rmtree(marks.parent)
        for leaf in recipe.reader.leaves(name, root, pin):
            if leaf.is_dir():
                v30.umask_modes(leaf)
    elif stage == "stats":
        leaves = recipe.reader.leaves(name, root, pin)
        stats.run(name, root, leaves, recipe.stats, subset)
        for step in recipe.horizon:
            step(name, root, leaves, subset, train)
    elif stage == "verify":
        leaves = recipe.reader.leaves(name, root, pin)
        verify.check_counts(name, leaves, pin, leaf_drops(name, root, subset, leaves)[0])
        command = (
            f"pixi run -e data python scripts/data/pretrain_data.py stats {name}"
            f"{' --subset' if subset else ''} --root {root}"
            f"{' --config configs/cpt.yaml' if recipe.horizon else ''}"
        )
        for leaf in leaves:
            verify.check_leaf(leaf, command)
        verify.check_euler(name, leaves)
        verify.dry_load(name, root, train)
        logger.info("%s: verified %d leaves", name, len(leaves))


def run_names(
    stage: str,
    names: list[str],
    root: Path,
    subset: bool,
    shard: tuple[int, int],
    train: TrainConfig,
    detect_again: bool = False,
) -> None:
    """One stage over ``names``; after curate, the count of episode keys they share."""
    keys: dict[str, set[dedup.Key]] = {}
    for name in names:
        logger.info("%s %s (root %s)", stage, name, root)
        if catalog.is_vl(name):
            run_vl(stage, name, root, subset, shard)
            if stage == "download":
                download.downloaded_marker(root, name, subset).touch()
        elif stage == "curate":
            keys[name] = curate(name, root, subset, detect_again)
        else:
            run_stage(stage, name, root, subset, shard, train)
            if stage == "download":
                download.downloaded_marker(root, name, subset).touch()
    if len(keys) > 1:
        logger.info("%d episode keys occur in more than one dataset", dedup.shared_across(keys))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument(
        "spec", metavar="name", help="dataset or group (docs/pretraining_data.md#build)"
    )
    parser.add_argument("--root", type=Path, help="datasets root (default $REKI_DATASETS_ROOT)")
    parser.add_argument(
        "--subset", action="store_true", help="build a small test part; use a separate --root"
    )
    parser.add_argument(
        "--shard",
        help="i/n: build part i of n (sharded datasets only, docs/pretraining_data.md#reference)",
    )
    parser.add_argument(
        "--detect", action="store_true", help="curate: make the lists of excluded episodes again"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="training config, for stats and verify of egodex, egoverse, xperience",
    )
    args = parser.parse_args(argv)
    if args.subset and args.root is None:
        parser.error("--subset writes to its own root: pass --root")
    if args.shard is not None and args.stage != "build":
        parser.error("--shard is a build option")
    names = resolve(args.spec)
    horizon = [n for n in names if catalog.recipes()[n].horizon]
    if args.stage in ("stats", "verify") and horizon and not args.config:
        given = " ".join(sys.argv[1:] if argv is None else argv)
        parser.error(
            f"{args.stage} of {horizon} needs the training config. Run:\n  pixi run -e data python "
            f"scripts/data/pretrain_data.py {given} --config configs/cpt.yaml"
        )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    download.quiet_http_logs()

    root = (args.root or datasets_root()).resolve()
    shard = parse_shard(args.shard or "0/1")
    train = TrainConfig.from_config_and_overrides(args.config, {})
    root.mkdir(parents=True, exist_ok=True)
    (root / "upstream" / ".downloaded").mkdir(parents=True, exist_ok=True)
    if args.stage == "build":
        missing = [
            n for n in names if not download.downloaded_marker(root, n, args.subset).exists()
        ]
        if missing:
            sys.exit(
                f"{', '.join(missing)}: not downloaded into {root / 'upstream'}. Run first:\n"
                f"  pixi run -e data python scripts/data/pretrain_data.py download {args.spec}"
                f"{' --subset' if args.subset else ''} --root {root}"
            )
    if args.stage == "download":
        sources = [s for n in names for s in catalog.pin(n, args.subset).sources]
        try:
            download.check_access(sources, root)
        except PermissionError as e:
            sys.exit(str(e))
    with tempfile.TemporaryDirectory(prefix=".tmp", dir=root) as tmp:
        tempfile.tempdir = tmp
        try:
            run_names(args.stage, names, root, args.subset, shard, train, args.detect)
        finally:
            tempfile.tempdir = None


if __name__ == "__main__":
    main()
