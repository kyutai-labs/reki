"""The pin of an upstream of the pretraining data: repo or URLs, revision, allow-patterns, and the
expected counts that ``build`` and ``verify`` check."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HfSnapshot:
    """A HF dataset repo at a fixed commit."""

    repo: str
    revision: str
    allow: tuple[str, ...] | None = None


@dataclass(frozen=True)
class HttpFiles:
    """Files fetched over HTTP; each URL maps to its expected ETag."""

    name: str
    etags: dict[str, str]
    untar: bool = False


@dataclass(frozen=True)
class HfHead:
    """The first ``nbytes`` of one file of a HF dataset repo at a fixed commit (a ``--subset``
    slice of a large archive), saved at its repo path."""

    repo: str
    revision: str
    path: str
    nbytes: int


Source = HfSnapshot | HttpFiles | HfHead


@dataclass(frozen=True)
class Pin:
    """One dataset's upstream and the counts it must give.

    ``episodes`` / ``frames`` are the totals of the episodes the reader reads (``select``: these
    upstream episode indices, None: all). ``subset`` is the ``--subset`` pin: narrower sources (e.g.
    ``meta/**`` and a few episode files), its ``select`` and counts; None: the subset is the whole
    dataset. Upstream meta totals are checked against the full pin. ``frames`` None: not
    checked (a reader whose frame total is known only after the build).
    """

    sources: tuple[Source, ...]
    episodes: int
    frames: int | None
    select: tuple[int, ...] | None = None
    subset: Pin | None = None
