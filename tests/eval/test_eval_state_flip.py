"""The eval state feed (``prep_eval_state``) reproduces what training feeds the state
conditioning: normalize, THEN negate the dataset's state gripper dims."""

import pytest

from reki.data.robotics.transforms import negate_gripper_dims

pytestmark = pytest.mark.cpu


def test_eval_state_feed_mirrors_the_train_state_flip(monkeypatch: pytest.MonkeyPatch):
    """The reference is composed from the training-side registry
    (`DatasetSpec.gripper_flip_state`), so the two sides diverging fails. Normalizing without
    the flip would show a state-conditioned checkpoint its gripper inverted against training."""
    import torch

    from reki.data.robotics.normalization import normalize_feature
    from reki.data.robotics.registry import DATASETS, DELTA_EE, DatasetSpec
    from reki.inference.features import prep_eval_state

    spec = DatasetSpec("Test", DELTA_EE, {}, "single_arm_ee_euler", gripper_flip_state=[7])
    monkeypatch.setitem(DATASETS, "state_flip", spec)
    stats = {"qk01": torch.full((8,), -1.0), "qk99": torch.ones(8)}
    raw = torch.linspace(-0.8, 0.8, 8)
    train_ref = normalize_feature(raw, stats)
    negate_gripper_dims(train_ref, spec.gripper_flip_state)

    got = prep_eval_state(raw, stats, dataset="state_flip")
    assert torch.equal(got, train_ref)
    unflipped = prep_eval_state(raw, stats, dataset="libero")
    assert torch.equal(unflipped[:7], got[:7])
    assert torch.equal(unflipped[7:], -got[7:]) and got[7:].abs().min() > 0
