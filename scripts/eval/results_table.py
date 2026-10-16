#!/usr/bin/env python3
"""Tabulate the rollout results under one or more eval output dirs (``reki.eval.results``).

python scripts/eval/results_table.py <out_dir> [<out_dir> ...] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from reki.eval.results import Table, collect, print_summary, summarize


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("out_dirs", nargs="+", type=Path)
    p.add_argument("--json", action="store_true", help="print the summary as JSON instead")
    args = p.parse_args(argv)
    merged: Table = defaultdict(lambda: defaultdict(list))
    for d in args.out_dirs:
        if not d.is_dir():
            print(f"{d}: not a directory", file=sys.stderr)
            return 2
        for bench, groups in collect(d).items():
            for group, units in groups.items():
                merged[bench][group].extend(units)
    if not merged:
        print("no leaf results.json found", file=sys.stderr)
        return 1
    summary = summarize(merged)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
