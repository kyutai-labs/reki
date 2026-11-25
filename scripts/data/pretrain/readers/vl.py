"""Vision-language (VL) co-training data of ``configs/cpt.yaml`` (``vl_datasets``): QA rows with
images, one dir per config, read by ``reki.data.utils.load_vl_config``. These are not LeRobot
leaves: ``curate`` and ``stats`` do nothing for them.

- ``finevision``, ``llava_onevision``: the configs of ``FINEVISION_FOR_ROBOTICS`` /
  ``LLAVA_EXT_FOR_ROBOTICS`` (``reki.data.robotics.vl``) from the HF parquet snapshots:
  ``build`` copies each config's rows, in order, to Arrow at
  ``<root>/vl/{finevision,llava_onevision}/<config>/``. A ``--subset`` build marks its root
  (``VL_SUBSET_MARKER``): the loader then reads only the configs built.
- ``robopoint``, ``refspatial``: QA JSON + image tarballs → Arrow at
  ``<root>/vl/spatial/<config>/``. Turns ``{from, value}`` become ``{role, content}``,
  ``<image>`` is stripped, coordinates [0, 1] go to [0, 1000]; RoboPoint turns are merged per
  image. Image bytes stream from the tarballs (no untar).
- ``pixmo``: PixMo-Points. ``download`` scrapes each image from its URL (sha256-checked); a
  failed URL drops the rows of its image. ``build`` synthesizes one QA per row.

Every config is ``data-NNNNN-of-MMMMM.arrow`` shards (Arrow IPC streams, the files
``datasets.Dataset.from_file`` memory-maps), read by ``reki.data.utils.load_vl_config`` with no
copy into ``$HF_DATASETS_CACHE``. The loader and ``verify`` also read a parquet config dir
(the loader with a copy); a dir holds one format. Built schema of the spatial configs:
``{id, images: list<Image>, conversations: list<{role, content}>, data_source}``.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import logging
import os
import re
import shutil
import tarfile
import time
import urllib.error
import urllib.request
import zlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from http.client import HTTPException
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from reki.data.robotics.vl import FINEVISION_FOR_ROBOTICS, VL_SUBSET_MARKER, llava_ext_configs
from reki.paths import FINEVISION_DIR, LLAVA_EXT_DIR, VL_SPATIAL_DIR

from ..common.download import ConcatFile, upstream_dir
from ..pins import HfHead, HfSnapshot, Pin, Source
from ..recipe import Recipe, VlReader

logger = logging.getLogger(__name__)

ROWS_PER_SHARD = 5000
ROWS_PER_BATCH = 100
"""Arrow record batch size, as ``datasets`` writes image datasets (random reads touch one)."""
SUBSET_ROWS = 300
SUBSET_HEAD = 400 << 20
"""``--subset``: the first ``SUBSET_ROWS`` rows of each config, from the first
``SUBSET_HEAD`` bytes of each tarball (PixMo: the first ``SUBSET_IMAGES`` images)."""
SUBSET_IMAGES = 400
SCRAPE_WORKERS = 64
SCRAPE_TIMEOUT_S = 8.0
SCRAPE_RETRIES = 2
_RETRY_HTTP = {429, 500, 502, 503, 504}
_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)


@dataclass(frozen=True)
class VlPin(Pin):
    """``rows``: the row count of each output config (None: known only after the scrape);
    ``limit``: at most this many rows per config (PixMo: images), the ``--subset`` slice.
    ``episodes`` is the total of ``rows``."""

    rows: dict[str, int | None] = field(default_factory=dict)
    limit: int | None = None


def vl_pin(
    sources: tuple[Source, ...],
    rows: dict[str, int | None],
    limit: int | None = None,
    subset: VlPin | None = None,
) -> VlPin:
    total = sum(n or 0 for n in rows.values())
    return VlPin(sources, total, None, subset=subset, rows=rows, limit=limit)


def features():
    """The ``datasets`` features of a built config."""
    from datasets import Features, Image, Value

    return Features(
        {
            "id": Value("string"),
            "images": [Image()],
            "conversations": [{"role": Value("string"), "content": Value("string")}],
            "data_source": Value("string"),
        }
    )


def write_shards(out: Path, schema: pa.Schema, shards: Iterable[Iterable[pa.RecordBatch]]) -> int:
    """Write each of ``shards`` (its record batches) to ``<out>/data-NNNNN-of-MMMMM.arrow``,
    replacing ``out`` (written in ``<out>.tmp``, then renamed); returns the row count."""
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    files: list[Path] = []
    n = 0
    for batches in shards:
        files.append(tmp / f"data-{len(files):05d}.arrow")
        with pa.OSFile(str(files[-1]), "wb") as sink, pa.ipc.new_stream(sink, schema) as writer:
            for batch in batches:
                writer.write_batch(batch)
                n += batch.num_rows
    for i, f in enumerate(files):
        f.rename(tmp / f"data-{i:05d}-of-{len(files):05d}.arrow")
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    return n


def write_config(out: Path, rows: Iterable[dict]) -> int:
    """Write ``rows`` (``features()``) as Arrow shards of ``ROWS_PER_SHARD`` rows to ``out``
    (no rows: one empty shard); returns the row count."""
    schema = features().arrow_schema
    shards = itertools.batched(rows, ROWS_PER_SHARD)
    first = next(shards, ())
    return write_shards(
        out,
        schema,
        (
            [
                pa.RecordBatch.from_pylist(list(b), schema=schema)
                for b in itertools.batched(shard, ROWS_PER_BATCH)
            ]
            or [pa.RecordBatch.from_pylist([], schema=schema)]
            for shard in itertools.chain([first], shards)
        ),
    )


def parquet_to_arrow(files: list[Path], out: Path) -> int:
    """Copy the rows of ``files`` in order (one shard per file, its schema and ``datasets``
    features metadata unchanged) to Arrow shards in ``out``; returns the row count."""
    schema = pq.read_schema(files[0])
    return write_shards(
        out, schema, (pq.ParquetFile(f).iter_batches(batch_size=ROWS_PER_BATCH) for f in files)
    )


def shard_files(out: Path) -> list[Path]:
    return sorted(out.glob("data-*.arrow"))


def read_shard(path: Path) -> pa.Table:
    """One Arrow shard or parquet file, memory-mapped."""
    if path.suffix == ".parquet":
        return pq.read_table(path, memory_map=True)
    return pa.ipc.open_stream(pa.memory_map(str(path))).read_all()


def shard_head(path: Path) -> tuple[pa.Schema, int]:
    """The schema and row count of an Arrow shard (memory-mapped) or a parquet file (footer)."""
    if path.suffix == ".parquet":
        meta = pq.ParquetFile(path)
        return meta.schema_arrow, meta.metadata.num_rows
    table = read_shard(path)
    return table.schema, table.num_rows


def config_files(out: Path, name: str) -> list[Path]:
    """The ``data-*.arrow`` shards or the ``*.parquet`` files of a config dir, which holds one
    format, as ``reki.data.utils.load_vl_config`` reads it."""
    files, parquet = shard_files(out), sorted(out.glob("*.parquet"))
    if parquet and (files or (out / "state.json").is_file() or (out / "0.0.0").is_dir()):
        raise ValueError(f"{out}: holds both parquet and Arrow files; keep one format per dir")
    if not files and not parquet:
        raise FileNotFoundError(
            f"{out}: no data-*.arrow or *.parquet files. Run:\n  "
            f"pixi run -e data python scripts/data/pretrain_data.py build {name}"
        )
    return files or parquet


def load_sample(out: str, root: Path, config: str) -> None:
    """Load the first row of ``config`` under ``<root>/<out>`` with its training loader: it has
    an image and a QA pair. The loader's path variable is set to that dir (a different value is
    reported: training reads that one)."""
    from reki.data.text.finevision import CachedArrowFineVision
    from reki.data.text.llavaov_onefive import LLava15OV
    from reki.data.text.vl_spatial_parquet import VLSpatialParquet

    var, loader = {
        FINEVISION_DIR: ("FINEVISION_PATH", CachedArrowFineVision),
        LLAVA_EXT_DIR: ("LLAVA_EXT_PATH", LLava15OV),
        VL_SPATIAL_DIR: ("VL_SPATIAL_PARQUET_PATH", VLSpatialParquet),
    }[out]
    path = str(root / out)
    if os.environ.get(var) not in (None, "", path):
        logger.warning("$%s is set: training reads %s, not %s", var, os.environ[var], path)
    os.environ[var] = path
    ds = loader(config, 1)
    item = ds.ds[0]
    images = item[ds.img_key]
    if not ds.prepare_for_formatter(item) or not (images if isinstance(images, list) else [images]):
        raise ValueError(f"{path}/{config}: row 0 has no QA pair or no image")


# ------------------------------------------------------------------ FineVision, LLaVA-OV


class Snapshot(VlReader):
    """A HF parquet snapshot of one dir per config: ``build`` copies each config to Arrow
    shards (a config already built is kept). ``columns``: the columns the loader reads;
    ``configs``: all configs of the full pin (a build of fewer is a subset: it writes
    ``VL_SUBSET_MARKER``)."""

    def __init__(self, out: str, columns: tuple[str, ...], configs: tuple[str, ...]) -> None:
        self.out, self.columns, self.configs = out, columns, configs

    def leaves(self, name: str, root: Path, pin: Pin) -> list[Path]:
        assert isinstance(pin, VlPin)
        return [root / self.out / c for c in pin.rows]

    def build(self, name: str, root: Path, pin: Pin, shard: tuple[int, int]) -> None:
        assert isinstance(pin, VlPin)
        up = upstream_dir(root, pin.sources[0])
        marker = root / self.out / VL_SUBSET_MARKER
        marker.parent.mkdir(parents=True, exist_ok=True)
        if set(pin.rows) == set(self.configs):
            marker.unlink(missing_ok=True)
        else:
            marker.write_text(json.dumps(sorted(pin.rows)))
        i, n = shard
        for out in self.leaves(name, root, pin)[i::n]:
            if out.is_symlink():
                out.unlink()
            if shard_files(out):
                logger.info("%s: %s already built", name, out.name)
                continue
            files = sorted((up / out.name).glob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"{up / out.name}: no parquet files; run download {name}")
            start = time.monotonic()
            rows = parquet_to_arrow(files, out)
            logger.info("%s: %d rows → %s (%.0f s)", out.name, rows, out, time.monotonic() - start)

    def verify(self, name: str, root: Path, pin: Pin) -> None:
        assert isinstance(pin, VlPin)
        for config, expected in pin.rows.items():
            files = config_files(root / self.out / config, name)
            heads = [shard_head(f) for f in files]
            missing = set(self.columns) - set(heads[0][0].names)
            if missing:
                raise KeyError(f"{files[0]}: no columns {sorted(missing)}")
            rows = sum(n for _, n in heads)
            if rows != expected:
                raise ValueError(f"{name}/{config}: {rows} rows, pinned {expected}")
        load_sample(self.out, root, min(pin.rows, key=lambda c: pin.rows[c] or 0))
        logger.info("%s: verified %d configs (%d rows)", name, len(pin.rows), pin.episodes)


# ------------------------------------------------------------------ RoboPoint, RefSpatial


@dataclass(frozen=True)
class ArchiveConfig:
    """One output config of a QA-JSON + image-tarball source, paths relative to the upstream
    dir. ``archives``: tarball globs (the sorted ``.part_*`` files of one glob are one
    stream); ``key``: how the JSON names an image of the tarball (``fullpath`` or
    ``basename``); ``group``: merge the turns of all entries of one image into one row
    (RoboPoint); ``prefix``: only images under this tarball dir."""

    jsons: tuple[str, ...]
    archives: tuple[str, ...]
    key: str
    group: bool
    prefix: str | None = None


@dataclass
class Entry:
    """One output row: ``id``, the image keys it needs, the turns."""

    id: str
    keys: list[str]
    turns: list[dict[str, str]]


def norm_turns(conversation: list[dict]) -> list[dict[str, str]]:
    """``[{from, value}]`` → ``[{role, content}]``, ``<image>`` stripped, coords [0,1]→[0,1000]."""
    from reki.data.text.coord_norm import coords_unit_to_1000

    return [
        {
            "role": "user" if t.get("from") == "human" else "assistant",
            "content": coords_unit_to_1000(re.sub(r"<image>", "", t.get("value", "")).strip()),
        }
        for t in conversation
    ]


def image_key(image: str, mode: str) -> str:
    return image if mode == "fullpath" else os.path.basename(str(image))


def entries(up: Path, cfg: ArchiveConfig) -> list[Entry]:
    """The rows of ``cfg``'s JSONs, in JSON order (grouped: in order of first occurrence)."""
    out: list[Entry] = []
    merged: dict[str, list[dict[str, str]]] = defaultdict(list)
    for path in cfg.jsons:
        for e in json.loads((up / path).read_text()):
            images = e.get("image")
            images = images if isinstance(images, list) else [images]
            images = [i for i in (images[:1] if cfg.group else images) if i]
            if not images or (
                cfg.prefix and not all(str(i).startswith(cfg.prefix) for i in images)
            ):
                continue
            turns = norm_turns(e.get("conversations", []))
            if not turns:
                continue
            keys = [image_key(i, cfg.key) for i in images]
            if cfg.group:
                merged[keys[0]].extend(turns)
            else:
                out.append(Entry(str(e.get("id") or keys[0]), keys, turns))
    return out + [Entry(k, [k], t) for k, t in merged.items()]


def tar_rows(
    up: Path, cfg: ArchiveConfig, rows: list[Entry], data_source: str, heads: set[str]
) -> Iterator[dict]:
    """Stream the tarballs once; yield each row when its last image arrives, then drop the
    bytes no pending row needs. A tarball that ends early stops the stream only when it is a
    pinned head (``heads``: upstream paths of ``HfHead`` sources)."""
    by_key: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        for k in row.keys:
            by_key[k].append(i)
    remaining = [set(row.keys) for row in rows]
    done = [False] * len(rows)
    data: dict[str, bytes] = {}
    for pattern in cfg.archives:
        parts = sorted(up.glob(pattern))
        if not parts:
            raise FileNotFoundError(f"{up}/{pattern}: no files; run download")
        try:
            with (
                io.BufferedReader(ConcatFile(parts), buffer_size=1 << 22) as reader,
                tarfile.open(fileobj=reader, mode="r|gz") as tar,
            ):
                for member in tar:
                    k = member.name if cfg.key == "fullpath" else os.path.basename(member.name)
                    if not member.isfile() or k not in by_key or k in data:
                        continue
                    f = tar.extractfile(member)
                    assert f is not None
                    data[k] = f.read()
                    for i in by_key[k]:
                        remaining[i].discard(k)
                        if not remaining[i] and not done[i]:
                            done[i] = True
                            yield {
                                "id": rows[i].id,
                                "images": [{"bytes": data[kk], "path": kk} for kk in rows[i].keys],
                                "conversations": rows[i].turns,
                                "data_source": data_source,
                            }
                    if all(done[i] for i in by_key[k]):
                        data[k] = b""
        except (EOFError, tarfile.ReadError, zlib.error):
            if not any(str(p.relative_to(up)) in heads for p in parts):
                raise
            logger.info("%s: end of the head of %s", data_source, parts[-1].name)


class Archive(VlReader):
    """QA JSONs + image tarballs → one parquet config per ``configs`` entry."""

    out = VL_SPATIAL_DIR

    def __init__(self, configs: dict[str, ArchiveConfig]) -> None:
        self.configs = configs

    def leaves(self, name: str, root: Path, pin: Pin) -> list[Path]:
        assert isinstance(pin, VlPin)
        return [root / self.out / c for c in pin.rows]

    def build(self, name: str, root: Path, pin: Pin, shard: tuple[int, int]) -> None:
        assert isinstance(pin, VlPin)
        up = upstream_dir(root, pin.sources[0])
        heads = {s.path for s in pin.sources if isinstance(s, HfHead)}
        i, n = shard
        for out in self.leaves(name, root, pin)[i::n]:
            cfg = self.configs[out.name]
            todo = entries(up, cfg)
            rows = itertools.islice(tar_rows(up, cfg, todo, out.name, heads), pin.limit)
            written = write_config(out, rows)
            logger.info("%s: %d rows of %d JSON entries → %s", out.name, written, len(todo), out)

    def verify(self, name: str, root: Path, pin: Pin) -> None:
        """Rows = the pinned count = the JSON entries (``limit``: the pinned count, all of
        them JSON entries), each with its id, turns and image paths; the built features."""
        assert isinstance(pin, VlPin)
        up = upstream_dir(root, pin.sources[0])
        for out in self.leaves(name, root, pin):
            todo = entries(up, self.configs[out.name])
            built: Counter[tuple[str, str, tuple[str, ...]]] = Counter()
            for f in config_files(out, name):
                table = read_shard(f)
                check_features(table.schema, f)
                built.update(
                    zip(
                        table.column("id").to_pylist(),
                        map(json.dumps, table.column("conversations").to_pylist()),
                        image_paths(table.column("images")),
                    )
                )
            rows, pinned = sum(built.values()), pin.rows[out.name]
            if rows != pinned or (pin.limit is None and rows != len(todo)):
                raise ValueError(
                    f"{out.name}: {rows} rows, {len(todo)} JSON entries, pinned {pinned}"
                )
            expected = Counter((e.id, json.dumps(e.turns), tuple(e.keys)) for e in todo)
            extra = built - expected
            if extra:
                raise ValueError(
                    f"{out.name}: {sum(extra.values())} rows not in the JSON, e.g. id "
                    f"{next(iter(extra))[0]}"
                )
        load_sample(self.out, root, min(pin.rows, key=lambda c: pin.rows[c] or 0))
        logger.info("%s: verified %d configs (%d rows)", name, len(pin.rows), pin.episodes)


def image_paths(images: pa.ChunkedArray) -> Iterator[tuple[str, ...]]:
    """Per row of a ``list<Image>`` column, its image paths (the bytes are not read)."""
    for chunk in images.chunks:
        assert isinstance(chunk, pa.ListArray)
        paths = iter(chunk.flatten().field("path").to_pylist())
        for k in chunk.value_lengths().to_pylist():
            yield tuple(itertools.islice(paths, k))


def check_features(schema: pa.Schema, path: Path) -> None:
    """The ``datasets`` features of the built shard ``path`` (its ``schema``) are
    ``features()``."""
    from datasets import Features

    got = Features.from_arrow_schema(schema)
    if got != features():
        raise ValueError(f"{path}: features {got}, want {features()}")


# ------------------------------------------------------------------ PixMo-Points


def pixmo_images(up: Path, limit: int | None) -> list[tuple[str, str]]:
    """``(sha256, url)`` of each unique image of the upstream rows, in row order; the first
    ``limit``."""
    import pandas as pd

    files = sorted((up / "data").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"{up}/data: no parquet files; run download pixmo")
    df = pd.concat([pd.read_parquet(f, columns=["image_sha256", "image_url"]) for f in files])
    df = df.drop_duplicates(subset="image_sha256")
    return list(zip(df["image_sha256"], df["image_url"]))[:limit]


def fetch_image(sha: str, url: str, out: Path) -> str | None:
    """Download one image to ``out/<sha>.jpg``, retried on a timeout, a connection error or
    HTTP 429/5xx. Returns None when it is on disk, else the failure."""
    dst = out / f"{sha}.jpg"
    if dst.exists():
        return None
    failure = "unknown"
    for attempt in range(SCRAPE_RETRIES + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(request, timeout=SCRAPE_TIMEOUT_S) as r:
                data = r.read()
            if hashlib.sha256(data).hexdigest() != sha:
                return "sha256 mismatch"
            tmp = dst.with_name(f"{dst.name}.{os.getpid()}.tmp")
            tmp.write_bytes(data)
            os.replace(tmp, dst)
            return None
        except urllib.error.HTTPError as e:
            failure = f"HTTP {e.code}"
            if e.code not in _RETRY_HTTP:
                return failure
        except (OSError, HTTPException, ValueError) as e:
            failure = type(e).__name__
        time.sleep(0.4 * (attempt + 1))
    return failure


def scrape(up: Path, limit: int | None) -> None:
    """Fetch the images of ``pixmo_images`` into ``<up>/images/``. Each failure is logged and
    listed in ``images/_failed.tsv`` (sha256, reason, URL); its rows are not built."""
    items = pixmo_images(up, limit)
    out = up / "images"
    out.mkdir(parents=True, exist_ok=True)
    failed: list[tuple[str, str, str]] = []
    logger.info("pixmo: %d images, %d workers", len(items), SCRAPE_WORKERS)
    with ThreadPoolExecutor(SCRAPE_WORKERS) as pool:
        futures = {pool.submit(fetch_image, sha, url, out): (sha, url) for sha, url in items}
        for i, future in enumerate(as_completed(futures), 1):
            reason = future.result()
            if reason is not None:
                sha, url = futures[future]
                failed.append((sha, reason, url))
                logger.info("pixmo: image %s failed (%s): %s", sha, reason, url)
            if i % 10000 == 0:
                logger.info("pixmo: %d/%d images, %d failed", i, len(items), len(failed))
    (out / "_failed.tsv").write_text("".join(f"{s}\t{r}\t{u}\n" for s, r, u in sorted(failed)))
    logger.info(
        "pixmo: %d images kept, %d failed (%.1f%%), listed in %s",
        len(items) - len(failed),
        len(failed),
        100 * len(failed) / max(1, len(items)),
        out / "_failed.tsv",
    )


def pixmo_turns(label: str, points: list[tuple[float, float]], count: int, method: str):
    """One synthesized QA, points in [0, 1000] (upstream: percent); absent → an abstention."""
    if method == "pointing":
        q = (
            f'Point to all "{label}" in the image. Answer with a list of points (x, y), '
            "each coordinate between 0 and 1000."
        )
        a = (
            "[" + ", ".join(f"({round(x * 10)}, {round(y * 10)})" for x, y in points) + "]"
            if points
            else f'There are no "{label}" in the image.'
        )
    else:
        q = f'How many "{label}" are in the image?'
        a = f'There are no "{label}" in the image.' if count == 0 else str(count)
    return [{"role": "user", "content": q}, {"role": "assistant", "content": a}]


def pixmo_qa(up: Path, kept: set[str]) -> Iterator[tuple[str, list[dict[str, str]]]]:
    """``(sha256, turns)`` of each upstream row whose image is in ``kept``, in row order."""
    import pandas as pd

    columns = ["image_sha256", "points", "count", "label", "collection_method"]
    for f in sorted((up / "data").glob("*.parquet")):
        df = pd.read_parquet(f, columns=columns)
        for sha, points, count, label, method in zip(*(df[c] for c in columns)):
            if sha not in kept:
                continue
            pts = [(float(p["x"]), float(p["y"])) for p in (points if points is not None else [])]
            n = len(pts) if count is None or pd.isna(count) else int(count)
            yield str(sha), pixmo_turns(str(label), pts, n, str(method))


class Pixmo(VlReader):
    """PixMo-Points: the rows of the images the scrape kept (``limit``: of the first images)."""

    out = VL_SPATIAL_DIR
    config = "pixmo_points"

    def leaves(self, name: str, root: Path, pin: Pin) -> list[Path]:
        return [root / self.out / self.config]

    def download(self, name: str, root: Path, pin: Pin) -> None:
        assert isinstance(pin, VlPin)
        super().download(name, root, pin)
        scrape(upstream_dir(root, pin.sources[0]), pin.limit)

    def kept(self, up: Path, pin: VlPin) -> set[str]:
        """The images on disk among the pin's images."""
        images = up / "images"
        return {sha for sha, _ in pixmo_images(up, pin.limit) if (images / f"{sha}.jpg").exists()}

    def build(self, name: str, root: Path, pin: Pin, shard: tuple[int, int]) -> None:
        assert isinstance(pin, VlPin)
        if shard[0] != 0:
            return
        up = upstream_dir(root, pin.sources[0])
        kept = self.kept(up, pin)

        def rows() -> Iterator[dict]:
            for sha, turns in pixmo_qa(up, kept):
                name = f"{sha}.jpg"
                yield {
                    "id": sha,
                    "images": [{"bytes": (up / "images" / name).read_bytes(), "path": name}],
                    "conversations": turns,
                    "data_source": self.config,
                }

        out = self.leaves(name, root, pin)[0]
        written = write_config(out, rows())
        logger.info("%s: %d rows of %d kept images → %s", self.config, written, len(kept), out)

    def verify(self, name: str, root: Path, pin: Pin) -> None:
        """The rows are those of the kept images (not of all upstream rows): ids and turns."""
        assert isinstance(pin, VlPin)
        up = upstream_dir(root, pin.sources[0])
        kept = self.kept(up, pin)
        if not kept:
            raise ValueError(f"{up}/images: no image kept; run download {name}")
        expected = Counter((sha, json.dumps(turns)) for sha, turns in pixmo_qa(up, kept))
        built: Counter[tuple[str, str]] = Counter()
        out = self.leaves(name, root, pin)[0]
        for f in config_files(out, name):
            table = read_shard(f)
            check_features(table.schema, f)
            built.update(
                zip(
                    table.column("id").to_pylist(),
                    map(json.dumps, table.column("conversations").to_pylist()),
                )
            )
        if built != expected:
            raise ValueError(
                f"{self.config}: {sum(built.values())} rows, the {len(kept)} kept images have "
                f"{sum(expected.values())} ({sum((built - expected).values())} not theirs)"
            )
        load_sample(self.out, root, self.config)
        logger.info("%s: verified %d rows of %d kept images", name, sum(built.values()), len(kept))


# ------------------------------------------------------------------ pins

FINEVISION = HfSnapshot("HuggingFaceM4/FineVision", "a890e0476b650030271ac4a083145610c08a7ee9")
LLAVA = HfSnapshot(
    "mvp-lab/LLaVA-OneVision-1.5-Instruct-Data", "0efb7ad47f8ba36e1d262545cadec4cd8960dc5f"
)
ROBOPOINT = HfSnapshot("wentao-yuan/robopoint-data", "7b06a4831afe7d5d46f19db0f0042ab4f58388e8")
REFSPATIAL = HfSnapshot("JingkunAn/RefSpatial", "519a6fd43aee2d0ed1366776e50f9456d212f94f")
PIXMO = HfSnapshot("allenai/pixmo-points", "2b5c6931e790e00ae00d4a2857e5f95d88f09a66", ("data/*",))

FINEVISION_ROWS: dict[str, int | None] = {
    "LLaVA_Instruct_150K": 157_710,
    "a_okvqa": 54_602,
    "alfworldgpt": 45_073,
    "allava_laion": 468_664,
    "allava_vflan": 177_078,
    "aokvqa": 16_539,
    "art": 5_492,
    "cambrian(filtered)_processed": 83_123,
    "clevr": 70_000,
    "clevr_math": 70_000,
    "coco_colors": 118_287,
    "cocoqa": 46_287,
    "cocotext": 16_169,
    "densefusion_1m": 1_058_751,
    "drivelm": 4_072,
    "est_vqa": 19_358,
    "google_landmarks": 299_993,
    "image_textualization(filtered)": 99_573,
    "indoor_qa": 3_350,
    "laion_gpt4v": 9_301,
    "lnqa": 302_780,
    "localized_narratives": 199_998,
    "lrv_normal(filtered)": 10_489,
    "lvis_instruct4v": 222_711,
    "mimic_cgd": 70_939,
    "mmevol": 160_215,
    "nlvr2": 50_426,
    "objects365_qa": 1_665_847,
    "ocrvqa": 165_746,
    "oodvqa": 8_488,
    "scienceqa": 4_976,
    "scienceqa(nona_context)": 19_208,
    "sharegpt4o": 57_284,
    "sharegpt4v(coco)": 50_017,
    "sharegpt4v(knowledge)": 1_988,
    "sharegpt4v(llava)": 29_986,
    "sharegpt4v(sam)": 8_990,
    "sketchyvqa": 8_000,
    "spatialsense": 10_440,
    "spot_the_diff": 8_566,
    "st_vqa": 17_247,
    "super_clevr(mathv360k)": 8_642,
    "tallyqa": 98_680,
    "textcaps": 21_906,
    "textocr(gpt4v)": 25_060,
    "textvqa": 21_943,
    "vision_flan(filtered)": 175_964,
    "visual7w": 14_366,
    "vqav2": 82_772,
    "vsr": 2_157,
    "wildvision": 333,
}
LLAVA_ROWS: dict[str, int | None] = {
    "gqa": 71_239,
    "llava_cot_100k": 98_229,
    "llava_wild": 54_240,
    "llrv_gpt4v": 13_820,
    "sherlock": 24_458,
    "svit-part-00-of-10": 98_516,
    "svit-part-01-of-10": 98_516,
    "svit-part-02-of-10": 98_516,
    "svit-part-03-of-10": 98_516,
    "svit-part-04-of-10": 98_516,
    "svit-part-05-of-10": 98_516,
    "svit-part-06-of-10": 98_516,
    "svit-part-07-of-10": 98_516,
    "svit-part-08-of-10": 98_516,
    "svit-part-09-of-10": 98_506,
    "viquae": 4_125,
    "vision_oritented": 14_262,
    "visual_chat": 25_903,
}
"""Rows per config at the pins (``FINEVISION_FOR_ROBOTICS``, ``llava_ext_configs``)."""
assert set(FINEVISION_ROWS) == FINEVISION_FOR_ROBOTICS and list(LLAVA_ROWS) == llava_ext_configs()

_RP_JSON = ("robopoint_1432k.json",)
ROBOPOINT_CONFIGS = {
    f"robopoint_{part}": ArchiveConfig(
        _RP_JSON, (f"images/{part}.tar.gz.part_*",), "fullpath", True, f"{part}/"
    )
    for part in ("object_ref", "region_ref")
}
_RS_2D = ("2D/image/image.tar.gz.part_*",)
_RS_3D = ("3D/image/image.tar.gz",)
REFSPATIAL_CONFIGS = {
    "refspatial_2d_choice": ArchiveConfig(("2D/choice_qa.json",), _RS_2D, "basename", False),
    "refspatial_2d_reasoning": ArchiveConfig(
        ("2D/reasoning_template_qa.json",), _RS_2D, "basename", False
    ),
    "refspatial_3d_choice": ArchiveConfig(("3D/choice_qa.json",), _RS_3D, "basename", False),
    "refspatial_3d_multi_view": ArchiveConfig(
        ("3D/multi_view_qa.json",),
        ("3D/image_multi_view/image_multi_view.tar.gz",),
        "basename",
        False,
    ),
    "refspatial_3d_reasoning": ArchiveConfig(
        ("3D/reasoning_template_qa.json",), _RS_3D, "basename", False
    ),
    "refspatial_3d_vacant": ArchiveConfig(("3D/vacant_qa.json",), _RS_3D, "basename", False),
    "refspatial_3d_visual_choice": ArchiveConfig(
        ("3D/visual_choice_qa.json",),
        ("3D/image_visual_choice/image_visual_choice.tar.gz.part_*",),
        "basename",
        False,
    ),
}


def _archive_pin(
    source: HfSnapshot,
    configs: dict[str, ArchiveConfig],
    heads: tuple[str, ...],
    rows: dict[str, int | None],
    subset_rows: dict[str, int | None],
) -> VlPin:
    """The full pin (JSONs + tarballs) and the subset pin (JSONs + the first ``SUBSET_HEAD``
    bytes of ``heads``, the first file of each tarball; ``SUBSET_ROWS`` rows per config)."""
    jsons = tuple(dict.fromkeys(j for c in configs.values() for j in c.jsons))
    archives = tuple(dict.fromkeys(a for c in configs.values() for a in c.archives))
    repo, revision = source.repo, source.revision
    return vl_pin(
        (HfSnapshot(repo, revision, jsons + archives),),
        rows,
        subset=vl_pin(
            (
                HfSnapshot(repo, revision, jsons),
                *(HfHead(repo, revision, h, SUBSET_HEAD) for h in heads),
            ),
            subset_rows,
            SUBSET_ROWS,
        ),
    )


PINS = {
    "finevision": vl_pin(
        (replace(FINEVISION, allow=tuple(f"{c}/*" for c in FINEVISION_ROWS)),),
        FINEVISION_ROWS,
        subset=vl_pin((replace(FINEVISION, allow=("vsr/*",)),), {"vsr": 2157}),
    ),
    "llava_onevision": vl_pin(
        (replace(LLAVA, allow=tuple(f"{c}/*" for c in LLAVA_ROWS)),),
        LLAVA_ROWS,
        subset=vl_pin((replace(LLAVA, allow=("viquae/*",)),), {"viquae": 4125}),
    ),
    "robopoint": _archive_pin(
        ROBOPOINT,
        ROBOPOINT_CONFIGS,
        ("images/object_ref.tar.gz.part_00", "images/region_ref.tar.gz.part_00"),
        {"robopoint_object_ref": 287_956, "robopoint_region_ref": 268_174},
        dict.fromkeys(ROBOPOINT_CONFIGS, SUBSET_ROWS),
    ),
    "refspatial": _archive_pin(
        REFSPATIAL,
        REFSPATIAL_CONFIGS,
        (
            "2D/image/image.tar.gz.part_a",
            "3D/image/image.tar.gz",
            "3D/image_multi_view/image_multi_view.tar.gz",
            "3D/image_visual_choice/image_visual_choice.tar.gz.part_a",
        ),
        {
            "refspatial_2d_choice": 460_657,
            "refspatial_2d_reasoning": 460_657,
            "refspatial_3d_choice": 216_412,
            "refspatial_3d_multi_view": 77_650,
            "refspatial_3d_reasoning": 351_897,
            "refspatial_3d_vacant": 231_554,
            "refspatial_3d_visual_choice": 539_976,
        },
        {**dict.fromkeys(REFSPATIAL_CONFIGS, SUBSET_ROWS), "refspatial_3d_multi_view": 22},
    ),
    "pixmo": vl_pin(
        (PIXMO,),
        {"pixmo_points": None},
        subset=vl_pin((PIXMO,), {"pixmo_points": None}, SUBSET_IMAGES),
    ),
}
"""RoboPoint/RefSpatial rows = their JSON entries (subset: 22 multi-view rows have all their
images in the head). PixMo rows depend on the images still online."""

RECIPES = {
    "finevision": Recipe(
        Snapshot(FINEVISION_DIR, ("images", "texts"), tuple(FINEVISION_ROWS)), curate=(), stats=()
    ),
    "llava_onevision": Recipe(
        Snapshot(LLAVA_EXT_DIR, ("image", "conversations"), tuple(LLAVA_ROWS)),
        curate=(),
        stats=(),
    ),
    "robopoint": Recipe(Archive(ROBOPOINT_CONFIGS), curate=(), stats=()),
    "refspatial": Recipe(Archive(REFSPATIAL_CONFIGS), curate=(), stats=()),
    "pixmo": Recipe(Pixmo(), curate=(), stats=()),
}

GROUPS = {"vl": list(RECIPES)}
"""The ``vl`` group: every VL dataset (``vl_datasets`` of ``configs/cpt.yaml``)."""
