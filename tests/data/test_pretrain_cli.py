"""Pins and command line of the pretraining pipeline (``scripts/data/pretrain_data.py``)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pretrain import catalog
from pretrain.common import download
from pretrain.pins import HfSnapshot

pytestmark = pytest.mark.cpu


def test_every_recipe_has_a_pinned_registry_dataset() -> None:
    from reki.data.robotics.registry import DATASETS

    assert set(catalog.recipes()) <= set(catalog.pins())
    for name, pin in catalog.pins().items():
        assert name in DATASETS or catalog.is_vl(name)
        for source in pin.sources:
            if isinstance(source, HfSnapshot):
                assert re.fullmatch(r"[0-9a-f]{40}", source.revision), name


def test_shard_and_spec_parsing() -> None:
    from scripts.data.pretrain_data import parse_shard, resolve

    assert parse_shard("2/4") == (2, 4)
    with pytest.raises(ValueError):
        parse_shard("4/4")
    assert resolve("berkeley_cable_routing") == ["berkeley_cable_routing"]
    with pytest.raises(SystemExit, match="'libero': no pretraining build. Groups: abc, egocentric"):
        resolve("libero")


def test_subset_needs_its_own_root() -> None:
    from scripts.data.pretrain_data import main

    with pytest.raises(SystemExit):
        main(["build", "berkeley_cable_routing", "--subset"])
    with pytest.raises(SystemExit):
        main(["curate", "berkeley_cable_routing", "--shard", "0/2"])


def test_build_needs_the_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.data.pretrain_data as cli

    with pytest.raises(SystemExit) as exit_:
        cli.main(["build", "berkeley_cable_routing", "--subset", "--root", str(tmp_path)])
    err = str(exit_.value.code)
    assert "berkeley_cable_routing: not downloaded" in err
    assert f"download berkeley_cable_routing --subset --root {tmp_path}" in err
    built = []
    monkeypatch.setattr(cli, "run_stage", lambda stage, name, *a: built.append((stage, name)))
    cli.main(["download", "berkeley_cable_routing", "--subset", "--root", str(tmp_path)])
    cli.main(["build", "berkeley_cable_routing", "--subset", "--root", str(tmp_path)])
    assert built == [("download", "berkeley_cable_routing"), ("build", "berkeley_cable_routing")]


def test_only_the_last_shard_finishes(tmp_path: Path) -> None:
    from scripts.data.pretrain_data import last_shard

    assert not last_shard(tmp_path, "abc", (1, 3))
    assert not last_shard(tmp_path, "abc", (0, 3))
    assert last_shard(tmp_path, "abc", (2, 3))
    assert not last_shard(tmp_path, "abc", (0, 3))
    assert not last_shard(tmp_path, "abc", (0, 2))


def test_horizon_stats_need_the_training_config(capsys: pytest.CaptureFixture[str]) -> None:
    from scripts.data.pretrain_data import main

    for stage in ("stats", "verify"):
        with pytest.raises(SystemExit):
            main([stage, "egodex", "--subset", "--root", "R"])
        err = capsys.readouterr().err
        assert f"{stage} egodex --subset --root R --config configs/cpt.yaml" in err


def test_http_request_logs_are_quiet(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    download.quiet_http_logs()
    with caplog.at_level(logging.INFO):
        logging.getLogger("httpx").info("HTTP Request: GET https://cdn/x?user_id=1")
        logging.getLogger("huggingface_hub.file_download").info("signed url")
        logging.getLogger("httpx").warning("kept")
    assert [r.getMessage() for r in caplog.records] == ["kept"]


def test_snapshot_check_rejects_another_revision(tmp_path: Path) -> None:
    meta = tmp_path / ".cache" / "huggingface" / "download" / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json.metadata").write_text("a" * 40 + "\netag\n1.0\n")
    download.check_snapshot(tmp_path, "a" * 40)
    with pytest.raises(ValueError, match="not at"):
        download.check_snapshot(tmp_path, "b" * 40)


def test_snapshot_lists_prefixes_once_and_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub
    from huggingface_hub.errors import HfHubHTTPError
    from huggingface_hub.hf_api import RepoFile, RepoFolder

    rev = "c" * 40
    calls: dict[str, list] = {"tree": [], "files": [], "get": []}

    class FakeApi:
        def list_repo_files(self, *_: object, **__: object) -> list[str]:
            calls["files"].append(1)
            return []

        def list_repo_tree(self, repo: str, path_in_repo: str, **_: object) -> list:
            calls["tree"].append(path_in_repo)
            names = {"meta": ["meta/info.json"], "data": ["data/a.parquet", "data/b.parquet"]}
            folder = RepoFolder(path=f"{path_in_repo}/sub", oid="x", tree_id="x")
            return [folder, *(RepoFile(path=n, size=1, oid="x") for n in names[path_in_repo])]

    class TooMany(HfHubHTTPError):
        def __init__(self) -> None:
            self.response = type("R", (), {"status_code": 429})()

    def fake_get(repo: str, name: str, local_dir: str, **_: object) -> str:
        calls["get"].append(name)
        if calls["get"].count(name) == 1 and name == "data/b.parquet":
            raise TooMany()
        (Path(local_dir) / name).parent.mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / name).touch()
        meta = Path(local_dir) / ".cache" / "huggingface" / "download" / f"{name}.metadata"
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text(f"{rev}\netag\n")
        return name

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_get)
    monkeypatch.setattr(download.time, "sleep", lambda _: None)
    source = HfSnapshot("org/repo", rev, ("meta/*", "data/**"))
    download.snapshot(source, tmp_path)
    download.snapshot(source, tmp_path)
    assert sorted(calls["tree"]) == ["data", "meta"] and not calls["files"]
    assert sorted(calls["get"]) == [
        "data/a.parquet",
        "data/b.parquet",
        "data/b.parquet",
        "meta/info.json",
    ]
    download._LISTINGS.clear()
    download.snapshot(source, tmp_path)  # the on-disk listing: no new tree call
    assert len(calls["tree"]) == 2
    assert download.is_complete(source, tmp_path)


def _fake_head(monkeypatch: pytest.MonkeyPatch, gated: set[str]) -> list[str]:
    """Patch the HF HEAD: 403 for the ``gated`` repos; returns the probed URLs."""
    import huggingface_hub
    from huggingface_hub.errors import GatedRepoError

    class Gated(GatedRepoError):
        def __init__(self) -> None:
            self.response = type("R", (), {"status_code": 403})()

    probed: list[str] = []

    def fake_head(url: str) -> object:
        probed.append(url)
        if any(f"/datasets/{repo}/" in url for repo in gated):
            raise Gated()
        return object()

    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", fake_head)
    return probed


def test_access_check_lists_every_gated_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_head(monkeypatch, {"org/b", "org/c"})
    sources = [HfSnapshot(f"org/{r}", "d" * 40) for r in "abc"]
    with pytest.raises(PermissionError) as e:
        download.check_access(sources, tmp_path)
    assert "https://huggingface.co/datasets/org/b\n" in str(e.value)
    assert "https://huggingface.co/datasets/org/c\n" in str(e.value)
    assert "org/a" not in str(e.value) and "click Agree" in str(e.value)


def test_access_check_passes_and_skips_complete_repos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probed = _fake_head(monkeypatch, set())
    sources = [HfSnapshot(f"org/{r}", "d" * 40, ("meta/*",)) for r in "abc"]
    marker = download.upstream_dir(tmp_path, sources[0]) / ".cache" / f"complete@{'d' * 40}.json"
    marker.parent.mkdir(parents=True)
    marker.write_text('[["meta/*"]]')
    download.check_access(sources, tmp_path)
    assert len(probed) == 2 and not any("/org/a/" in url for url in probed)
