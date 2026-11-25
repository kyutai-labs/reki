"""Load one vision-language config dir from disk."""

from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk


def load_vl_config(root: str, cfg: str) -> Dataset:
    """Load ``<root>/<cfg>``, which holds one format:

    - Arrow, memory-mapped with no copy: ``data-*.arrow`` shards (written by
      ``scripts/data/pretrain_data.py build``), a ``save_to_disk`` dir (``state.json``), or an
      HF builder cache (``0.0.0/<hash>/fine_vision-train*.arrow``);
    - parquet (``*.parquet``, e.g. a Hub snapshot), through ``load_dataset``: the first load
      writes an Arrow copy into ``$HF_DATASETS_CACHE``, later loads reuse it.

    A dir with both formats, or with neither, raises."""
    path = Path(root) / cfg
    parquet = sorted(path.glob("*.parquet"))
    shards = sorted(path.glob("data-*.arrow"))
    disk = (path / "state.json").is_file()
    builder = path / "0.0.0"
    if parquet and (shards or disk or builder.is_dir()):
        raise ValueError(
            f"{path}: holds both parquet and Arrow files; keep one format per config dir"
        )
    if parquet:
        return load_dataset("parquet", data_files=[str(p) for p in parquet], split="train")
    if disk:
        return load_from_disk(str(path))
    if shards:
        return concatenate_datasets([Dataset.from_file(str(a)) for a in shards])
    if builder.is_dir():
        hashes = [d for d in builder.iterdir() if d.is_dir()]
        if len(hashes) != 1:
            raise FileNotFoundError(f"{builder}: expected one hash dir, found {len(hashes)}")
        arrows = sorted(hashes[0].glob("fine_vision-train*.arrow"))
        if arrows:
            return concatenate_datasets([Dataset.from_file(str(a)) for a in arrows])
    raise FileNotFoundError(
        f"{path}: no *.parquet, data-*.arrow, state.json or "
        "0.0.0/<hash>/fine_vision-train*.arrow. Build it with "
        "scripts/data/pretrain_data.py build (the VL datasets)"
    )
