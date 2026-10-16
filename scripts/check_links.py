"""Check relative links and #anchors in every tracked .md file.

    pixi run python scripts/check_links.py

Covers markdown links/images ``[t](target)`` and HTML ``href=``/``src=``. Skips URLs with a
scheme and fenced code blocks. Anchors use GitHub's heading slugs. Exits 1 on a broken link.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(
    r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)|(?:href|src)=\"([^\"]+)\""
)
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")


def strip_code(text: str) -> list[str]:
    """The input lines with fenced and inline code removed."""
    lines, in_fence = [], False
    for line in text.splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
            lines.append("")
            continue
        lines.append("" if in_fence else re.sub(r"`[^`]*`", "", line))
    return lines


def slug(heading: str) -> str:
    """The GitHub-style anchor slug for a Markdown heading."""
    h = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)  # link text only
    h = re.sub(r"<[^>]+>", "", h).replace("`", "").strip().lower()
    h = re.sub(r"[^\w\- ]", "", h)
    return h.replace(" ", "-")


def anchors(path: Path) -> set[str]:
    """The GitHub-style heading and explicit HTML anchors in a Markdown file."""
    seen: dict[str, int] = {}
    out: set[str] = set()
    in_fence = False
    for line in path.read_text().splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        m = None if in_fence else HEADING.match(line)
        if m:
            s = slug(m.group(2))
            n = seen.get(s, 0)
            seen[s] = n + 1
            out.add(s if n == 0 else f"{s}-{n}")
    out |= set(re.findall(r"<a\s+(?:name|id)=\"([^\"]+)\"", path.read_text()))
    return out


def main() -> int:
    """Check tracked Markdown files for broken relative links and anchors."""
    files = subprocess.run(
        ["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    files = [f for f in files if (ROOT / f).exists()]
    bad = []
    for f in files:
        src = ROOT / f
        for lineno, line in enumerate(strip_code(src.read_text()), 1):
            for m in LINK.finditer(line):
                target = m.group(1) or m.group(2)
                if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target):
                    continue
                path_part, _, anchor = target.partition("#")
                dest = (src.parent / path_part).resolve() if path_part else src
                if not dest.exists():
                    bad.append(f"{f}:{lineno}: missing {target}")
                elif anchor and dest.suffix == ".md" and anchor not in anchors(dest):
                    bad.append(f"{f}:{lineno}: no anchor #{anchor} in {dest.relative_to(ROOT)}")
    print("\n".join(bad) or f"OK: {len(files)} files, no broken link")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
