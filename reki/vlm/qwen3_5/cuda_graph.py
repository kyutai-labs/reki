"""Shared pieces of the manual inference CUDA graphs (``prefill_graph``, ``expert_graph``):
a checked capture and a small LRU of captured graphs."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable
from typing import Generic, TypeVar

import torch

Out = TypeVar("Out", bound=torch.Tensor | tuple[torch.Tensor, ...])
Entry = TypeVar("Entry")


def capture_checked(
    forward: Callable[[], Out], reference: tuple[torch.Tensor, ...], what: str
) -> tuple[torch.cuda.CUDAGraph, Out]:
    """Warm ``forward`` up twice on a side stream, capture it, replay once and check that the
    replay is bitwise equal to ``reference`` (the output of the uncaptured run). ``forward``
    returns a tensor or a tuple of tensors, matched against ``reference``."""
    side = torch.cuda.Stream()
    assert isinstance(side, torch.cuda.Stream)
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            forward()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = forward()
    graph.replay()
    got_all = out if isinstance(out, tuple) else (out,)
    for got, ref in zip(got_all, reference, strict=True):
        if not torch.equal(got, ref):
            diff = (got.float() - ref.float()).abs().max().item()
            raise RuntimeError(f"{what} graph differs from the {what} (max |Δ| {diff})")
    return graph, out


class GraphLRU(Generic[Entry]):
    """The ``max_entries`` most recently used captured graphs, by key.

    A graph reads the model's weights and buffers at their capture-time addresses: call
    ``clear()`` after replacing parameter tensors (``load_state_dict(assign=True)``, a dtype or
    device move, a module swap), or replay reads freed memory."""

    def __init__(self, max_entries: int) -> None:
        self.max_entries = max_entries
        self.graphs: OrderedDict[Hashable, Entry] = OrderedDict()

    def get(self, key: Hashable) -> Entry | None:
        entry = self.graphs.get(key)
        if entry is not None:
            self.graphs.move_to_end(key)
        return entry

    def make_room(self) -> None:
        """Evict before a capture, so the new graph can reuse the freed memory pool."""
        while len(self.graphs) >= self.max_entries:
            # Dropping the last reference frees the graph and its private memory pool.
            self.graphs.popitem(last=False)

    def put(self, key: Hashable, entry: Entry) -> None:
        self.graphs[key] = entry

    def clear(self) -> None:
        self.graphs.clear()
