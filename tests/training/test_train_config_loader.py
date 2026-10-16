"""``TrainConfig.from_config_and_overrides``: the ``--config`` yaml + CLI override path shared by
scripts/train.py and scripts/dry_run.py."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest
import sft
import sft_3n
import sft_4n
import yaml

from reki.policy.train_config import TrainConfig

pytestmark = pytest.mark.cpu

REPO = Path(__file__).resolve().parents[2]

SFT_LAUNCHERS = (sft, sft_3n, sft_4n)


def test_yaml_plus_override_merge(tmp_path: Path) -> None:
    yml = tmp_path / "c.yaml"
    yml.write_text(
        yaml.safe_dump(
            {"dataset": ["libero"], "steps": 10, "lr": 1e-4, "action_expert_num_registers": 32}
        )
    )
    cfg = TrainConfig.from_config_and_overrides(yml, {"steps": 3, "output_dir": str(tmp_path)})
    assert cfg.dataset == ["libero"]
    assert cfg.steps == 3
    assert cfg.lr == 1e-4
    assert cfg.output_dir == str(tmp_path)


@pytest.mark.parametrize("flag", ["help", "h"])
def test_help_lists_fields(flag: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    yml = tmp_path / "c.yaml"
    yml.write_text(yaml.safe_dump({"steps": 7}))
    with pytest.raises(SystemExit) as exc:
        TrainConfig.from_config_and_overrides(yml, {flag: True})
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "  steps: 7" in out
    assert "unify_ee_euler_rotation: True" in out


def test_unknown_yaml_key_raises(tmp_path: Path) -> None:
    yml = tmp_path / "c.yaml"
    yml.write_text(yaml.safe_dump({"steps": 10, "not_a_field": False}))
    with pytest.raises(ValueError, match="not_a_field"):
        TrainConfig.from_config_and_overrides(yml, {})
    with pytest.raises(ValueError, match="not_a_field"):
        TrainConfig.from_yaml(yml)


@pytest.mark.parametrize(
    ("launcher", "benchmark"),
    [(m, b) for m in SFT_LAUNCHERS for b in sorted(m.BENCHMARKS)],
    ids=lambda v: v if isinstance(v, str) else v.__name__,
)
def test_sft_launcher_cli_round_trip(launcher: ModuleType, benchmark: str) -> None:
    """The launcher's overrides survive the command line: launcheon formats them, the shell
    splits them, fire parses them back to the same values train.py receives."""
    import shlex

    import fire
    from common import FlagEqualsValue, as_cli
    from launcheon.utils import format_kwargs

    overrides = launcher.overrides(benchmark)
    tokens = shlex.split(format_kwargs(as_cli(overrides), FlagEqualsValue()))
    parsed = fire.Fire(lambda **kw: kw, command=tokens)
    dataset = parsed.pop("dataset")
    expected = dict(overrides)
    got_dataset = list(dataset) if isinstance(dataset, tuple) else [dataset]
    assert got_dataset == str(expected.pop("dataset")).split(",")
    assert parsed == expected


def test_shipped_configs_strict_load() -> None:
    # cpt.yaml (reference recipe of kyutai/reki-2b) keeps pretraining keys TrainConfig does not define
    for yml in sorted((REPO / "configs").glob("*.yaml")):
        if yml.name != "cpt.yaml":
            TrainConfig.load_yaml_kwargs(yml)


def test_checkpoint_config_rejects_python_objects(tmp_path: Path) -> None:
    (tmp_path / "train_config.yaml").write_text("steps: !!python/object/apply:os.getcwd []\n")
    with pytest.raises(yaml.constructor.ConstructorError):
        TrainConfig.from_checkpoint(tmp_path)


def test_bool_override_from_fire_string() -> None:
    cfg = TrainConfig.from_config_and_overrides(
        None, {"shuffle_cameras": "false", "ddp_static_graph": 1}
    )
    assert cfg.shuffle_cameras is False
    assert cfg.ddp_static_graph is True
    with pytest.raises(ValueError, match="shuffle_cameras"):
        TrainConfig.from_config_and_overrides(None, {"shuffle_cameras": "submit"})
