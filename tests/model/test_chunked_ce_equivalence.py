"""``packed_shifted_ce`` (the CE path of ``RekiFlowModel.forward``) on a 3-sample packed stream
equals each sample's own unchunked next-token CE: no label crosses a sample boundary, and
chunking + checkpointing change neither the values nor the gradients."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from reki.policy.modeling import IGNORE_INDEX, packed_shifted_ce

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize("checkpoint", [True, False])
def test_packed_chunked_ce_matches_per_sample_ce(checkpoint: bool) -> None:
    torch.manual_seed(0)
    lens, D, V = [5, 9, 7], 16, 50
    head = nn.Linear(D, V, bias=False)
    hidden = torch.randn(sum(lens), D, requires_grad=True)
    labels = torch.randint(0, V, (sum(lens),))
    labels[[0, 1, 6, 15]] = IGNORE_INDEX
    seq_lens = torch.tensor(lens)

    ce, _ = packed_shifted_ce(hidden, labels, seq_lens, head, checkpoint=checkpoint, chunk=4)

    expected, start = [], 0
    for i, n in enumerate(lens):
        h, lbl = hidden[start : start + n], labels[start : start + n]
        expected.append(
            F.cross_entropy(head(h[:-1]), lbl[1:], ignore_index=IGNORE_INDEX, reduction="none")
        )
        if i < len(lens) - 1:
            expected.append(torch.zeros(1))
        start += n
    ref = torch.cat(expected)
    torch.testing.assert_close(ce, ref, rtol=0, atol=1e-6)

    (g,) = torch.autograd.grad(ce.sum(), hidden)
    (g_ref,) = torch.autograd.grad(ref.sum(), hidden)
    torch.testing.assert_close(g, g_ref, rtol=0, atol=1e-6)
