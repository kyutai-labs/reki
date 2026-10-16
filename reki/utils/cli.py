"""Shared command-line argument parsers."""

import argparse

_BOOL_WORDS = {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}


def str2bool(s: str) -> bool:
    """``true/1/yes`` or ``false/0/no`` (any case) as a bool."""
    if s.lower() not in _BOOL_WORDS:
        raise argparse.ArgumentTypeError(f"expected true/false, got {s!r}")
    return _BOOL_WORDS[s.lower()]


def fire_bool(name: str, value: object) -> bool:
    """A python-fire flag as a bool: fire reads ``--x=false`` as the string ``"false"``."""
    if isinstance(value, bool):
        return value
    if str(value).lower() not in _BOOL_WORDS:
        raise ValueError(f"--{name} takes True or False, got {value!r}: write --{name}=True")
    return _BOOL_WORDS[str(value).lower()]
