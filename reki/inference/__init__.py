"""Inference: a checkpoint as a stateful policy. ``policy``: ``RekiPolicy``, the public API.
``spec``: the inference spec and stats of a checkpoint, read from its export
(``scripts/export_checkpoint.py``) or built from its finetuning dataset. ``features``: raw
observation → model batch, model action → native action. ``benchmarks``: the simulation presets.
"""
