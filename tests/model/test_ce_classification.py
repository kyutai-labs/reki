"""The CE regions of a packed batch (``classify_tokens``) and their weighting (``ce_losses``): a
robot sample's subtask (assistant turn or user stream) is robot CE, a VL sample's labels are
text CE, a robot sample without a subtask has no CE, and robot and text samples weigh by sample
count, not token count."""

from __future__ import annotations

import pytest
import torch

from reki.policy.constants import IGNORE_INDEX, RegionTag
from reki.policy.modeling import ce_losses, classify_tokens, packed_shifted_ce

pytestmark = pytest.mark.cpu

U, S, AH, US, UH = (
    RegionTag.USER,
    RegionTag.SUBTASK,
    RegionTag.ASST_HEADER,
    RegionTag.USER_SUBTASK,
    RegionTag.USER_SUBTASK_HEADER,
)
SUPERVISED = (S, US)


def _sample(tags: list[RegionTag], vl_labels: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """(labels, tags) of one sample: supervised where the tag is, or on the last ``vl_labels``
    tokens of a VL sample."""
    t = torch.tensor(tags, dtype=torch.int8)
    labels = torch.full((len(tags),), IGNORE_INDEX)
    sup = torch.isin(t, torch.tensor(SUPERVISED, dtype=torch.int8))
    if vl_labels:
        sup[-vl_labels:] = True
    labels[sup] = 7
    return labels, t


SAMPLES = [
    (_sample([U, U, U, AH, S, S, S]), True),  # robot, assistant-turn subtask
    (_sample([U, U, U, UH, US, US]), True),  # robot, user-stream subtask
    (_sample([U, U, U, U]), True),  # robot, no subtask
    (_sample([U, U, U, U, U, U], vl_labels=3), False),  # VL
]


def _packed(n: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[bool]]:
    """(labels, tags, seq_lens, is_robot_sample) of the first ``n`` of ``SAMPLES``."""
    samples = SAMPLES[:n]
    labels = torch.cat([lbl for (lbl, _), _ in samples])
    tags = torch.cat([t for (_, t), _ in samples])
    seq_lens = torch.tensor([len(lbl) for (lbl, _), _ in samples])
    return labels, tags, seq_lens, [robot for _, robot in samples]


def _shift(labels: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
    hidden = torch.zeros(labels.shape[0], 4)
    head = torch.nn.Linear(4, 10)
    return packed_shifted_ce(hidden, labels, seq_lens, head, checkpoint=False)[1]


def test_regions_of_a_mixed_batch() -> None:
    labels, tags, seq_lens, is_robot = _packed()
    shift_labels = _shift(labels, seq_lens)
    tb = classify_tokens(shift_labels, seq_lens, is_robot, region_tags=tags)
    label_tags, valid = tags[1:], shift_labels != IGNORE_INDEX
    sample = torch.repeat_interleave(torch.arange(4), seq_lens)[:-1]
    supervised = torch.isin(label_tags, torch.tensor(SUPERVISED, dtype=torch.int8))
    assert torch.equal(tb.subtask_region, supervised & valid)
    assert torch.equal(tb.robot_mask, valid & (sample < 2))
    assert torch.equal(tb.text_mask, valid & (sample == 3))
    assert (tb.n_robot_samples, tb.n_text_samples) == (2, 1)
    assert int(tb.robot_mask.sum()) == 5 and int(tb.text_mask.sum()) == 3


def test_vl_only_batch_without_tags_is_all_text() -> None:
    samples = [_sample([U] * 5, vl_labels=2), _sample([U] * 4, vl_labels=4)]
    labels = torch.cat([lbl for lbl, _ in samples])
    seq_lens = torch.tensor([5, 4])
    shift_labels = _shift(labels, seq_lens)
    tb = classify_tokens(shift_labels, seq_lens, [False, False])
    assert not tb.robot_mask.any() and not tb.subtask_region.any()
    assert torch.equal(tb.text_mask, shift_labels != IGNORE_INDEX)
    # The first label of a sample has no previous token in it: 2 + 3 supervised targets.
    assert int(tb.text_mask.sum()) == 5 and (tb.n_robot_samples, tb.n_text_samples) == (0, 2)


def test_robot_and_text_samples_weigh_by_sample_count() -> None:
    labels, tags, seq_lens, is_robot = _packed()
    shift_labels = _shift(labels, seq_lens)
    tb = classify_tokens(shift_labels, seq_lens, is_robot, region_tags=tags)
    ce = torch.rand(shift_labels.shape[0], generator=torch.Generator().manual_seed(0))
    out = ce_losses(ce, tb)
    robot, text = ce[tb.robot_mask].mean(), ce[tb.text_mask].mean()
    torch.testing.assert_close(out["ce_loss"], (2 * robot + 1 * text) / 3, rtol=0, atol=0)
    torch.testing.assert_close(out["text_loss"], text, rtol=0, atol=0)
    torch.testing.assert_close(out["subtask_loss"], ce[tb.subtask_region].mean(), rtol=0, atol=0)

    labels, tags, seq_lens, is_robot = _packed(3)
    shift_labels = _shift(labels, seq_lens)
    tb = classify_tokens(shift_labels, seq_lens, is_robot, region_tags=tags)
    out = ce_losses(ce[: shift_labels.shape[0]], tb)
    assert "text_loss" not in out
    torch.testing.assert_close(out["ce_loss"], ce[: shift_labels.shape[0]][tb.robot_mask].mean())
