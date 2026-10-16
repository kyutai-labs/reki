"""The text prompt of a robot sample. ``reki.inference.features`` builds the eval prompt with
the same functions."""

from __future__ import annotations

MIRROR_MARKER = "[MIRROR WORLD] "
"""Prefix of the task text of a mirrored sample."""


def normalize_task(s: str) -> str:
    """Ensure task string is capitalized and ends with a period."""
    s = s.strip()
    if not s:
        return s
    s = s[0].upper() + s[1:]
    if not s.endswith((".", "!", "?")):
        s += "."
    return s


def clean_task_text(task_text: str) -> str:
    """The text stripped and with underscores and newlines replaced by spaces."""
    return task_text.strip().replace("_", " ").replace("\n", " ")


def structured_prompt(
    task_text: str,
    robot_type: str,
    fps: float,
    mode: str | None = None,
    speed_bin: int | None = None,
    drop_embodiment: bool = False,
    drop_fps: bool = False,
) -> str:
    """The task prompt with one ``field: value`` per line: embodiment, instruction, speed, fps
    and control mode (joint or delta end-effector actions). Embodiment and fps can be dropped."""
    lines: list[str] = []
    if not drop_embodiment:
        lines.append(f"embodiment: {robot_type}")
    lines.append(f"instruction: {clean_task_text(task_text)}")
    if speed_bin is not None:
        lines.append(f"speed: {speed_bin}")
    if not drop_fps:
        lines.append(f"fps: {round(fps)}")
    if mode:
        lines.append(f"control mode: {mode}")
    return "\n".join(lines) + "\n"
