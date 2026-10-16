#!/usr/bin/env python3
"""Serve eval videos + results in a web browser.

Point it at any eval directory (a suite results dir, a `checkpoint-N` dir, or any parent).
It finds every task (a dir with a `videos/` subdir), reads the sibling `results.json`, and
serves a page of per-episode video players with success/fail badges.

    python3 scripts/eval/viewer.py <eval_path> [--port 8000] [--host 127.0.0.1]

Stdlib only; HTTP Range support so mp4 seeking works.
"""

import argparse
import html
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def discover(root: str):
    """Return {suite_rel: {task_name: {results, videos:[(name, rel, ok)]}}}."""
    suites = {}
    for dirpath, _, _ in os.walk(root):
        if os.path.basename(dirpath) != "videos":
            continue
        task_dir = os.path.dirname(dirpath)
        task_name = os.path.basename(task_dir)
        suite_rel = os.path.relpath(os.path.dirname(task_dir), root)
        results = None
        rp = os.path.join(task_dir, "results.json")
        if os.path.exists(rp):
            with open(rp) as f:
                results = json.load(f)
        by_ep = {e["episode"]: e for e in (results or {}).get("episodes", [])}
        vids = []
        for fn in sorted(os.listdir(dirpath)):
            if not fn.endswith(".mp4"):
                continue
            ep = by_ep.get(_ep_num(fn))
            ok = bool(ep["success"]) if ep is not None else None
            rel = os.path.relpath(os.path.join(dirpath, fn), root)
            vids.append({"fn": fn, "rel": rel, "ok": ok, "ep": ep})
        if not vids:
            continue
        suites.setdefault(suite_rel, {})[task_name] = {"results": results, "videos": vids}
    return suites


def _ep_num(name: str):
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 0


def render_index(root: str, suites: dict[str, Any]):
    """The HTML index for discovered evaluation suites and videos."""
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>eval viewer</title><style>",
        "body{font-family:system-ui,sans-serif;margin:0;background:#111;color:#eee}",
        "header{padding:16px 20px;background:#1b1b1b;border-bottom:1px solid #333;position:sticky;top:0;z-index:2}",
        "header code{color:#9cf;font-size:12px}",
        "h1{margin:0 0 4px;font-size:18px}",
        "details{border-bottom:1px solid #2a2a2a}",
        "summary{cursor:pointer;padding:12px 20px;font-size:16px;font-weight:600;user-select:none}",
        "summary:hover{background:#1b1b1b}",
        ".suite{padding:0 20px 12px}",
        ".task{margin:14px 0;padding:12px;background:#181818;border-radius:8px}",
        ".task h3{margin:0 0 8px;font-size:15px;display:flex;align-items:center;gap:10px}",
        ".rate{font-size:13px;font-weight:500;color:#bbb}",
        ".meta{font-size:12px;color:#9a9a9a;margin:-2px 0 10px;line-height:1.5}",
        ".meta b{color:#cfe;font-weight:600}",
        ".info{font-size:11px;color:#999;margin-top:3px}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}",
        ".cell{background:#222;border-radius:6px;padding:6px}",
        ".cell video{width:100%;border-radius:4px;display:block;background:#000}",
        ".cap{font-size:12px;margin-top:5px;display:flex;justify-content:space-between;align-items:center}",
        ".badge{font-size:11px;padding:2px 7px;border-radius:10px;font-weight:600}",
        ".ok{background:#164;color:#8f8}",
        ".no{background:#511;color:#f99}",
        ".un{background:#333;color:#ccc}",
        "</style></head><body>",
        f"<header><h1>eval viewer</h1><code>{html.escape(root)}</code></header>",
    ]
    if not suites:
        parts.append("<p style='padding:20px'>No task videos found under this path.</p>")
    for suite_rel in sorted(suites):
        tasks = suites[suite_rel]
        label = suite_rel if suite_rel != "." else os.path.basename(root)
        n_tasks = len(tasks)
        parts.append(
            f"<details open><summary>{html.escape(label)} <span class='rate'>({n_tasks} task{'s' if n_tasks != 1 else ''})</span></summary><div class='suite'>"
        )
        for task_name in sorted(tasks):
            info = tasks[task_name]
            res = info["results"] or {}
            rate = (
                f"{res['success_rate'] * 100:.0f}% ({res['n_success']}/{res['n_episodes']})"
                if res
                else ""
            )
            parts.append(f"<div class='task'><h3>{html.escape(task_name)}")
            if rate:
                parts.append(f"<span class='rate'>{rate}</span>")
            parts.append("</h3>")
            meta = []
            for k in ("n_action_steps", "target_episodes"):
                if res.get(k) is not None:
                    meta.append(f"{k}={html.escape(str(res[k]))}")
            if meta:
                parts.append(f"<div class='meta'>{' · '.join(meta)}</div>")
            parts.append("<div class='grid'>")
            for v in sorted(info["videos"], key=lambda v: _ep_num(v["fn"])):
                src = "/video?path=" + html.escape(v["rel"], quote=True)
                ok = v["ok"]
                badge = (
                    "<span class='badge ok'>success</span>"
                    if ok
                    else "<span class='badge no'>fail</span>"
                    if ok is False
                    else "<span class='badge un'>—</span>"
                )
                ep = v["ep"]
                info_bits = [f"steps {ep['steps']}"] if ep else []
                parts.append(
                    f"<div class='cell'><video preload='none' controls muted loop src='{src}'></video>"
                    f"<div class='cap'><span>{html.escape(v['fn'])}</span>{badge}</div>"
                    f"<div class='info'>{' · '.join(info_bits)}</div></div>"
                )
            parts.append("</div></div>")
        parts.append("</div></details>")
    parts.append("</body></html>")
    return "".join(parts).encode()


class Handler(BaseHTTPRequestHandler):
    """Serve the evaluation index and seekable video files."""

    root: str | None = None

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            assert self.root is not None
            body = render_index(self.root, discover(self.root))
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/video?path="):
            from urllib.parse import unquote

            rel = unquote(self.path[len("/video?path=") :])
            self._serve_video(rel)
            return
        self.send_error(404)

    def _serve_video(self, rel: str):
        assert self.root is not None
        full = os.path.realpath(os.path.join(self.root, rel))
        if not (full == self.root or full.startswith(self.root + os.sep)) or not os.path.isfile(
            full
        ):
            self.send_error(404)
            return
        size = os.path.getsize(full)
        ctype = "video/mp4"
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng and rng.startswith("bytes="):
            s, _, e = rng[6:].partition("-")
            start = int(s) if s else 0
            end = int(e) if e else size - 1
            end = min(end, size - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        length = end - start + 1
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        with open(full, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    break
                remaining -= len(chunk)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="eval directory (suite / checkpoint / higher)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    root = os.path.realpath(args.path)
    if not os.path.isdir(root):
        ap.error(f"not a directory: {root}")
    suites = discover(root)
    n_tasks = sum(len(t) for t in suites.values())
    n_vids = sum(len(i["videos"]) for t in suites.values() for i in t.values())

    Handler.root = root
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"eval-viewer: {len(suites)} suite(s), {n_tasks} task(s), {n_vids} video(s)")
    print(f"  root: {root}")
    print(f"  serving on http://{args.host}:{args.port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
