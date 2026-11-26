"""Batch keys and token labels shared by the data pipeline and the model. The keys are the
strings of ``lerobot.utils.constants``: the data pipeline builds batches from LeRobot datasets
under these names."""

from enum import IntEnum

ACTION = "action"
OBS_STATE = "observation.state"
OBS_LANGUAGE_TOKENS = "observation.language.tokens"
ACTION_IS_PAD = "action_is_pad"
ACTION_DIM_VALID = "action_dim_valid"
"""``[T, D]`` bool, False on action dims without a target (untracked ego hands); absent = all valid."""

IGNORE_INDEX = -100
"""Label of a token that is not supervised."""


class RegionTag(IntEnum):
    """Per-token tag of a robot sample (``region_tags``, int8). Every tag except ``USER``,
    ``ASST_HEADER`` and ``USER_SUBTASK_HEADER`` is supervised. The action expert attends to the
    user turn (``USER``, ``USER_SUBTASK``, ``USER_SUBTASK_HEADER``)."""

    USER = 0
    """The user turn: images, prompt. Also every token of a sample without actions."""
    SUBTASK = 2
    """Subtask text in the assistant turn."""
    ASST_HEADER = 5
    """Opening tokens of the assistant (subtask) turn."""
    USER_SUBTASK = 6
    """Subtask text in the user turn (``subtask_in_user_stream``)."""
    USER_SUBTASK_HEADER = 7
    """Opening tokens of the subtask in the user turn (``subtask_in_user_stream``)."""
