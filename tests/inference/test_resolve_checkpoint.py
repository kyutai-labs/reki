"""``resolve_checkpoint``: a missing local path names the path; only ``org/name`` goes to the
Hub, and a Hub miss says that neither exists."""

from pathlib import Path

import pytest

from reki.policy.checkpoint import resolve_checkpoint

pytestmark = pytest.mark.cpu


@pytest.fixture
def hub_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    import httpx
    import huggingface_hub
    from huggingface_hub.errors import RepositoryNotFoundError

    calls: list[str] = []

    def snapshot_download(repo_id: str, **kwargs: object) -> str:
        calls.append(repo_id)
        response = httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co"))
        raise RepositoryNotFoundError("404", response=response)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    return calls


@pytest.mark.parametrize(
    "arg", ["checkpoints/reki-2b-libero", "./reki-2b-libero", "a/b/c", "reki-2b-libero"]
)
def test_missing_local_path(
    arg: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hub_calls: list[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "checkpoints").mkdir()
    with pytest.raises(FileNotFoundError, match=f"not found: {arg}"):
        resolve_checkpoint(arg)
    assert hub_calls == []


def test_hub_miss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hub_calls: list[str]) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="is not a local folder and is not a model id"):
        resolve_checkpoint("kyutai/reki-2b-nope")
    assert hub_calls == ["kyutai/reki-2b-nope"]
