"""Fetch the pinned upstreams into ``<root>/upstream/``: HF snapshots at a commit, the head of a
HF file, HTTP files with an ETag check, tar extraction, HTTP range reads. Any changed commit or
ETag raises."""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import tarfile
import threading
import time
import urllib.request
from collections.abc import Callable, Sequence
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar
from urllib.parse import urlsplit

from ..pins import HfHead, HfSnapshot, HttpFiles, Source

if TYPE_CHECKING:
    import requests

# hf_xet stalls on some files (0-byte .incomplete for minutes); plain HTTP downloads do not.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

logger = logging.getLogger(__name__)
RATE_LIMIT_RETRIES = 6
T = TypeVar("T")


def quiet_http_logs() -> None:
    """Keep the per-request INFO lines of the HTTP clients out of the log: they print signed CDN
    URLs that carry the HF user id. ``snapshot`` logs its own progress."""
    for name in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


def downloaded_marker(root: Path, name: str, subset: bool) -> Path:
    """Written when ``download`` of ``name`` ends; ``build`` needs it."""
    return root / "upstream" / ".downloaded" / f"{name}{'.subset' if subset else ''}"


def upstream_dir(root: Path, source: Source) -> Path:
    """Where ``source`` is downloaded under the datasets root ``root``."""
    name = source.name if isinstance(source, HttpFiles) else source.repo
    return root / "upstream" / name


def fetch(source: Source, out: Path) -> None:
    """Download one source into ``out``."""
    if isinstance(source, HfSnapshot):
        snapshot(source, out)
    elif isinstance(source, HfHead):
        head(source, out)
    else:
        http(source, out)


def _rate_limited(fn: Callable[[], T], what: str) -> T:
    """``fn()``, retried with backoff on HTTP 429 (the API limit shared by everyone on the
    token)."""
    import requests
    from huggingface_hub.errors import HfHubHTTPError

    for attempt in range(RATE_LIMIT_RETRIES):
        try:
            return fn()
        except (HfHubHTTPError, requests.HTTPError) as e:
            status = e.response.status_code if e.response is not None else None
            if status != 429 or attempt == RATE_LIMIT_RETRIES - 1:
                raise
            logger.warning("%s: HTTP 429, retry in %d s", what, 60 * 2**attempt)
            time.sleep(60 * 2**attempt)
    raise AssertionError


def _prefixes(allow: tuple[str, ...] | None) -> tuple[str, ...] | None:
    """The directories to list when every allow pattern is ``<dir>/*``, ``<dir>/**`` or a plain
    file path; None: the whole repo is listed."""
    if allow is None:
        return None
    out = []
    for pattern in allow:
        head, _, tail = pattern.rpartition("/")
        if not any(c in pattern for c in "*?["):
            out.append(head)
        elif tail in ("*", "**") and head and not any(c in head for c in "*?["):
            out.append(head)
        else:
            return None
    return tuple(dict.fromkeys(out))


_LISTINGS: dict[tuple[str, str, tuple[str, ...] | None], list[str]] = {}


def listing(source: HfSnapshot, out: Path) -> list[str]:
    """The repo files at the revision under the allow patterns' directories (or all), listed
    once per process and cached on disk in ``out/.cache/listing@<revision>.json``."""
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    prefixes = _prefixes(source.allow)
    key = (source.repo, source.revision, prefixes)
    if key in _LISTINGS:
        return _LISTINGS[key]
    cache = out / ".cache" / f"listing@{source.revision}.json"
    cached = json.loads(cache.read_text()) if cache.exists() else {}
    entry = "*" if prefixes is None else "\n".join(prefixes)
    if entry not in cached:
        api = HfApi()
        if prefixes is None:
            cached[entry] = _rate_limited(
                lambda: api.list_repo_files(
                    source.repo, repo_type="dataset", revision=source.revision
                ),
                source.repo,
            )
        else:
            cached[entry] = [
                f.path
                for prefix in prefixes
                for f in _rate_limited(
                    lambda prefix=prefix: list(
                        api.list_repo_tree(
                            source.repo,
                            path_in_repo=prefix or None,
                            repo_type="dataset",
                            revision=source.revision,
                            recursive=True,
                        )
                    ),
                    f"{source.repo}/{prefix}",
                )
                if isinstance(f, RepoFile)
            ]
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(cached))
    _LISTINGS[key] = cached[entry]
    return cached[entry]


def _at_revision(out: Path, name: str, revision: str) -> bool:
    meta = out / ".cache" / "huggingface" / "download" / f"{name}.metadata"
    return (out / name).exists() and meta.exists() and meta.read_text().startswith(revision)


def snapshot(source: HfSnapshot, out: Path) -> None:
    """Download every repo file under ``allow`` at the pinned revision (one ``hf_hub_download``
    per missing file, 8 in parallel, HTTP 429 retried), then check that all are on disk and
    came from the revision."""
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import hf_hub_download

    allow = source.allow
    wanted = [f for f in listing(source, out) if allow is None or any(fnmatch(f, p) for p in allow)]
    todo = [f for f in wanted if not _at_revision(out, f, source.revision)]
    logger.info("%s: %d files, %d to download", source.repo, len(wanted), len(todo))

    def fetch_one(name: str) -> None:
        _rate_limited(
            lambda: hf_hub_download(
                source.repo,
                name,
                repo_type="dataset",
                revision=source.revision,
                local_dir=str(out),
            ),
            f"{source.repo}/{name}",
        )

    with ThreadPoolExecutor(8) as pool:
        for done, _ in enumerate(pool.map(fetch_one, todo), 1):
            if done % 50 == 0 or done == len(todo):
                logger.info("%s: %d/%d downloaded", source.repo, done, len(todo))
    missing = [f for f in wanted if not (out / f).exists()]
    if missing:
        raise FileNotFoundError(f"{out}: {len(missing)} files missing, e.g. {missing[0]}")
    check_snapshot(out, source.revision)
    if not is_complete(source, out):
        marker = _complete_marker(source, out)
        done = json.loads(marker.read_text()) if marker.exists() else []
        marker.write_text(json.dumps([*done, _allow_key(source)]))


def _complete_marker(source: HfSnapshot, out: Path) -> Path:
    return out / ".cache" / f"complete@{source.revision}.json"


def _allow_key(source: HfSnapshot) -> list[str] | None:
    return None if source.allow is None else list(source.allow)


def is_complete(source: HfSnapshot, out: Path) -> bool:
    """Whether ``snapshot`` already downloaded ``source`` into ``out`` (its
    ``.cache/complete@<revision>.json`` lists the allow patterns)."""
    marker = _complete_marker(source, out)
    return marker.exists() and _allow_key(source) in json.loads(marker.read_text())


def head(source: HfHead, out: Path) -> None:
    """Download the first ``source.nbytes`` of ``source.path`` at the pinned revision into
    ``out/<path>`` (one range request, HTTP 429 retried); a file of that size is kept."""
    import requests
    from huggingface_hub import get_token, hf_hub_url

    dst = out / source.path
    if dst.exists() and dst.stat().st_size == source.nbytes:
        return
    url = hf_hub_url(source.repo, source.path, repo_type="dataset", revision=source.revision)
    token = get_token()
    headers = {"Range": f"bytes=0-{source.nbytes - 1}"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    logger.info("%s: first %d MB of %s", source.repo, source.nbytes >> 20, source.path)
    part = dst.with_name(dst.name + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(RATE_LIMIT_RETRIES):
        with requests.get(url, headers=headers, stream=True, timeout=60) as r:
            if r.status_code == 429 and attempt < RATE_LIMIT_RETRIES - 1:
                logger.warning("%s: HTTP 429, retry in %d s", source.repo, 60 * 2**attempt)
                time.sleep(60 * 2**attempt)
                continue
            r.raise_for_status()
            if r.status_code != 206:
                raise ValueError(f"{url}: no range support (HTTP {r.status_code})")
            with open(part, "wb") as f:
                for chunk in r.iter_content(8 << 20):
                    f.write(chunk)
            break
    if part.stat().st_size != source.nbytes:
        raise ValueError(f"{url}: got {part.stat().st_size} bytes, want {source.nbytes}")
    part.rename(dst)


def _done(source: HfSnapshot | HfHead, root: Path) -> bool:
    out = upstream_dir(root, source)
    if isinstance(source, HfHead):
        dst = out / source.path
        return dst.exists() and dst.stat().st_size == source.nbytes
    return is_complete(source, out)


def no_access(sources: Sequence[Source], root: Path) -> list[str]:
    """The repos of the HF ``sources`` not done under ``root`` whose files the token cannot
    download: a HEAD on ``.gitattributes`` at the pinned revision (8 in parallel, HTTP 429
    retried) gets a gated (403) or unauthorized (401) answer. A missing ``.gitattributes``
    (404) passes: the gate answers first."""
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError

    todo = {
        (s.repo, s.revision)
        for s in sources
        if isinstance(s, HfSnapshot | HfHead) and not _done(s, root)
    }

    def denied(repo_revision: tuple[str, str]) -> bool:
        repo, revision = repo_revision
        url = hf_hub_url(repo, ".gitattributes", repo_type="dataset", revision=revision)
        try:
            _rate_limited(lambda: get_hf_file_metadata(url), repo)
        except EntryNotFoundError:
            pass
        except HfHubHTTPError as e:
            if e.response is None or e.response.status_code not in (401, 403):
                raise
            return True
        return False

    with ThreadPoolExecutor(8) as pool:
        flags = list(pool.map(denied, sorted(todo)))
    logger.info("access probe: %d HF repos, %d not accessible", len(todo), sum(flags))
    return sorted({repo for (repo, _), bad in zip(sorted(todo), flags) if bad})


def check_access(sources: Sequence[Source], root: Path) -> None:
    """Raise ``PermissionError`` listing every repo of ``no_access`` with its page."""
    repos = no_access(sources, root)
    if repos:
        pages = "\n".join(f"  {r}  https://huggingface.co/datasets/{r}" for r in repos)
        raise PermissionError(
            f"{len(repos)} HF repos are not accessible with the current token:\n{pages}\n"
            "Open each page, click Agree; access is granted automatically for auto-gated repos."
        )


def check_snapshot(out: Path, revision: str) -> None:
    """Raise unless every file of the local snapshot ``out`` was downloaded at ``revision``
    (the commit in HF's ``.cache/huggingface/download/*.metadata``)."""
    metadata = sorted((out / ".cache" / "huggingface" / "download").rglob("*.metadata"))
    if not metadata:
        raise FileNotFoundError(f"{out}: no HF download metadata")
    stale = [m for m in metadata if m.read_text().split("\n", 1)[0] != revision]
    if stale:
        raise ValueError(f"{out}: {len(stale)} files not at {revision}, e.g. {stale[0]}")


CHANGED = (
    "(changed upstream after the release, so it cannot be built as released; please open an "
    "issue at https://github.com/kyutai-labs/reki/issues)"
)


def _etag(url: str) -> str:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request) as response:
        return response.headers["ETag"].strip('"')


def http(source: HttpFiles, out: Path) -> None:
    """Fetch each URL into ``out`` (resumable), after checking its ETag; untar if pinned so."""
    out.mkdir(parents=True, exist_ok=True)
    for url, etag in source.etags.items():
        got = _etag(url)
        if got != etag:
            raise ValueError(f"{url}: ETag {got}, pinned {etag} {CHANGED}")
        dst = out / url.rsplit("/", 1)[-1]
        if not dst.exists():
            part = dst.with_name(dst.name + ".part")
            start = part.stat().st_size if part.exists() else 0
            request = urllib.request.Request(url, headers={"Range": f"bytes={start}-"})
            with urllib.request.urlopen(request) as response, open(part, "ab") as f:
                shutil.copyfileobj(response, f)
            part.rename(dst)
        if source.untar:
            untar(dst, out)


def umask_filter(
    member_filter: Callable[[tarfile.TarInfo, str], tarfile.TarInfo | None] = tarfile.data_filter,
) -> Callable[[tarfile.TarInfo, str], tarfile.TarInfo | None]:
    """``member_filter``, then each file and dir gets the mode a plain ``open`` / ``mkdir``
    gives under the process umask: tarballs often store 0600, which a group cannot read."""
    mask = os.umask(0)
    os.umask(mask)

    def apply(member: tarfile.TarInfo, dest: str) -> tarfile.TarInfo | None:
        kept = member_filter(member, dest)
        if kept is None or not (kept.isreg() or kept.isdir()):
            return kept
        return kept.replace(mode=(0o777 if kept.isdir() else 0o666) & ~mask, deep=False)

    return apply


def untar(
    archive: Path,
    out: Path,
    fresh: bool = False,
    member_filter: Callable[[tarfile.TarInfo, str], tarfile.TarInfo | None] = tarfile.data_filter,
) -> None:
    """Extract ``archive`` into ``out`` once (a ``.<name>.done`` marker records it), modes under
    the umask; ``fresh``: every time, into an emptied ``out``."""
    done = out / f".{archive.name}.done"
    if fresh:
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
    elif done.exists():
        return
    with tarfile.open(archive) as tf:
        tf.extractall(out, filter=umask_filter(member_filter))
    done.touch()


class ConcatFile(io.RawIOBase):
    """The byte concatenation of ``paths`` as one read-only stream (``cat``)."""

    def __init__(self, paths: list[Path]) -> None:
        self.paths = list(paths)
        self.file = open(self.paths.pop(0), "rb")

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:  # type: ignore[override]
        while True:
            n = self.file.readinto(buffer)
            if n or not self.paths:
                return n or 0
            self.file.close()
            self.file = open(self.paths.pop(0), "rb")

    def close(self) -> None:
        self.file.close()
        super().close()


# ------------------------------------------------------------------ HTTP range reads

_SESSIONS = threading.local()


def _session():
    import requests

    if getattr(_SESSIONS, "s", None) is None:
        _SESSIONS.s = requests.Session()
    return _SESSIONS.s


def _checked(response: requests.Response) -> requests.Response:
    response.raise_for_status()
    return response


def _reconnecting(fn: Callable[[], T], what: str) -> T:
    """``_rate_limited(fn)``, retried with backoff when the connection breaks."""
    import requests

    for attempt in range(RATE_LIMIT_RETRIES):
        try:
            return _rate_limited(fn, what)
        except (requests.ConnectionError, requests.exceptions.ChunkedEncodingError) as e:
            if attempt == RATE_LIMIT_RETRIES - 1:
                raise
            logger.warning("%s: %s, retry in %d s", what, type(e).__name__, 2**attempt)
            time.sleep(2**attempt)
    raise AssertionError


class RangeFile(io.RawIOBase):
    """A read-only file over HTTP range requests, cached by page. One keep-alive session per
    thread (a fresh TLS handshake per file is the bottleneck on the HF CDN). The HEAD resolves
    the redirects once: the pages are read from the final URL, without ``headers`` when it is on
    another host (a signed CDN URL). HTTP 429 and broken connections are retried. ``etag`` pins
    the remote file: a different ETag raises."""

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        etag: str | None = None,
        page: int = 8 << 20,
    ) -> None:
        self.page = page
        head = _reconnecting(
            lambda: _checked(_session().head(url, headers=headers or {}, allow_redirects=True)),
            url,
        )
        if etag is not None and head.headers["ETag"].strip('"') != etag:
            raise ValueError(f"{url}: ETag {head.headers['ETag']}, pinned {etag} {CHANGED}")
        self.source, self.url = url, head.url
        same_host = urlsplit(head.url).netloc == urlsplit(url).netloc
        self.headers = (headers or {}) if same_host else {}
        self.size = int(head.headers["Content-Length"])
        self.pos = 0
        self.cache: dict[int, bytes] = {}

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        base = (0, self.pos, self.size)[whence]
        self.pos = base + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def readinto(self, buffer) -> int:  # type: ignore[override]
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def read(self, n: int | None = -1) -> bytes:
        end = self.size if n is None or n < 0 else min(self.pos + n, self.size)
        out = bytearray()
        while self.pos < end:
            page = self.pos // self.page
            if page not in self.cache:
                start = page * self.page
                stop = min(start + self.page, self.size) - 1
                headers = {**self.headers, "Range": f"bytes={start}-{stop}"}
                self.cache[page] = _reconnecting(
                    lambda: _checked(_session().get(self.url, headers=headers)), self.source
                ).content
            buf = self.cache[page]
            offset = self.pos - page * self.page
            take = min(len(buf) - offset, end - self.pos)
            out += buf[offset : offset + take]
            self.pos += take
        return bytes(out)
