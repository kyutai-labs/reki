#!/usr/bin/env python
"""Download the raw inputs of the released recipe (SFT benchmarks).

    pixi run --frozen python scripts/data/download.py --list
    pixi run --frozen python scripts/data/download.py <dataset> [--out <dir>] [--dry_run]

`--out` is the dataset's raw-input path (docs/data.md). The default is the path that `--list`
prints, under $REKI_DATASETS_ROOT.
`--only` selects a subset where the dataset supports it (`--list` says what it selects).
Every fetch resumes: files already on disk are skipped. Run on a node with internet.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tarfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

from reki.paths import datasets_root


@dataclass(frozen=True)
class Snapshot:
    """One `snapshot_download` call into `--out`."""

    repo_id: str
    allow: list[str] | None = None
    ignore: list[str] | None = None
    revision: str | None = None
    max_workers: int = 8

    def describe(self) -> str:
        rev = f"@{self.revision}" if self.revision else ""
        return (
            f"snapshot {self.repo_id}{rev} allow={self.allow} "
            f"ignore={self.ignore} max_workers={self.max_workers}"
        )

    def run(self, out: Path) -> None:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=str(out),
            allow_patterns=self.allow,
            ignore_patterns=self.ignore,
            max_workers=self.max_workers,
        )


@dataclass(frozen=True)
class Custom:
    """A step with its own logic; `text` is what `--dry_run` prints."""

    text: str
    fn: Callable[[Path], None]

    def describe(self) -> str:
        return self.text

    def run(self, out: Path) -> None:
        self.fn(out)


Step = Snapshot | Custom


def _robocasa365_todo(out: Path, manifest: Path, prefix: str) -> dict[Path, str]:
    """Latest-date tar URL per task under `prefix`, minus tasks already extracted."""
    entries: dict[str, str] = json.loads(manifest.read_text())
    latest: dict[Path, tuple[str, str]] = {}
    for key, url in entries.items():
        if not key.startswith(prefix + "/"):
            continue
        parts = key.split("/")
        if len(parts) != 5:  # <task>/mg/demo/...: machine-generated demos, not Human300
            continue
        split, category, task, date, _ = parts
        task_dir = out / split / category / task
        if task_dir not in latest or date > latest[task_dir][0]:
            latest[task_dir] = (date, url)
    if not latest:
        raise ValueError(f"no manifest key under prefix {prefix!r}")
    return {
        d: url
        for d, (_, url) in latest.items()
        if not (d / "lerobot" / "meta" / "info.json").exists()
    }


def _robocasa365_fetch(task_dir: Path, shared_url: str) -> None:
    base, shared_id = shared_url.rstrip("/").split("/s/")
    url = f"{base}/shared/static/{shared_id}.tar"
    task_dir.mkdir(parents=True, exist_ok=True)
    tar_path = task_dir / "lerobot.tar.part"
    for attempt in range(1, 4):
        try:
            subprocess.run(
                ["curl", "-sSL", "--fail", "--retry", "2", "-C", "-", "-o", str(tar_path), url],
                check=True,
            )
            with tarfile.open(tar_path) as tf:
                tf.extractall(task_dir, filter="data")
            break
        except (subprocess.CalledProcessError, tarfile.TarError):
            tar_path.unlink(missing_ok=True)
            if attempt == 3:
                raise
    if not (task_dir / "lerobot" / "meta" / "info.json").exists():
        raise RuntimeError(f"{task_dir}: extracted tar has no lerobot/meta/info.json")
    tar_path.unlink()
    print(f"  done {task_dir}", flush=True)


def _download_robocasa365(out: Path, manifest: Path, prefixes: list[str], workers: int) -> None:
    todo: dict[Path, str] = {}
    for prefix in prefixes:
        todo |= _robocasa365_todo(out, manifest, prefix)
    print(f"{len(todo)} tasks to download", flush=True)
    failed = []
    with ThreadPoolExecutor(workers) as ex:
        futures = {ex.submit(_robocasa365_fetch, d, url): d for d, url in todo.items()}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001 — keep going, raise the full list at the end
                print(f"FAILED {futures[fut]}: {e}", flush=True)
                failed.append(str(futures[fut]))
    if failed:
        raise RuntimeError(f"{len(failed)} downloads failed (re-run to resume): {failed}")


def _plan_robocasa365(args: argparse.Namespace) -> list[Step]:
    if args.manifest is None or args.only is None:
        raise ValueError(
            "robocasa365 needs --manifest <checkout>/robocasa/models/assets/box_links/"
            "box_links_ds.json and --only <split>/<category> [...]"
        )
    workers = args.max_workers or 8
    n = sum(len(_robocasa365_todo(args.out, args.manifest, p)) for p in args.only)
    return [
        Custom(
            f"box tar x{n} (latest date per task under {args.only}) -> "
            f"<out>/<split>/<category>/<Task>/lerobot, workers={workers}",
            lambda out: _download_robocasa365(out, args.manifest, args.only, workers),
        )
    ]


@dataclass(frozen=True)
class Entry:
    """A dataset download registry entry."""

    layout: str
    plan: Callable[[argparse.Namespace], list[Step]]
    only: str | None = None


def _snapshot(*steps: Snapshot) -> Callable[[argparse.Namespace], list[Step]]:
    return lambda args: list(steps)


REGISTRY: dict[str, Entry] = {
    "libero": Entry("HuggingFaceVLA/libero", _snapshot(Snapshot("HuggingFaceVLA/libero"))),
    "robocasa365": Entry(
        "robocasa365_atomic_human_raw",
        _plan_robocasa365,
        only="manifest key prefixes <split>/<category> (required); also needs --manifest",
    ),
}


def _print_registry() -> None:
    for name, e in REGISTRY.items():
        print(f"{name:20s} {e.layout:45s} {e.only and f'--only: {e.only}' or ''}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", nargs="?", choices=list(REGISTRY))
    parser.add_argument("--list", action="store_true", help="print the registry and exit")
    parser.add_argument("--out", type=Path, help="destination directory (default: the --list path)")
    parser.add_argument("--dry_run", action="store_true", help="print the planned fetches")
    parser.add_argument("--only", nargs="+", help="subset selection (see --list)")
    parser.add_argument("--max_workers", type=int, help="override the per-dataset worker count")
    parser.add_argument("--manifest", type=Path, help="robocasa365: box_links_ds.json")
    args = parser.parse_args()

    if args.list:
        _print_registry()
        return
    if args.dataset is None:
        parser.error("<dataset> is required (or --list)")
    entry = REGISTRY[args.dataset]
    if args.out is None:
        args.out = datasets_root() / entry.layout
    if args.only is not None:
        if entry.only is None:
            parser.error(f"{args.dataset} takes no --only")
    if args.manifest is not None and args.dataset != "robocasa365":
        parser.error("--manifest is robocasa365-only")

    steps = entry.plan(args)
    if args.max_workers is not None:
        steps = [
            replace(s, max_workers=args.max_workers) if isinstance(s, Snapshot) else s
            for s in steps
        ]
    print(f"{args.dataset} -> {args.out}")
    for i, s in enumerate(steps, 1):
        print(f"  [{i}/{len(steps)}] {s.describe()}")
    if args.dry_run:
        return
    args.out.mkdir(parents=True, exist_ok=True)
    for s in steps:
        s.run(args.out)
    print(f"done: {args.out}")


if __name__ == "__main__":
    main()
