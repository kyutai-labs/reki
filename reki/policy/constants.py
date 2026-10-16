"""Batch keys shared by the data pipeline and the model. The keys are the strings of
``lerobot.utils.constants``: the data pipeline builds batches from LeRobot datasets under
these names."""

ACTION = "action"
OBS_STATE = "observation.state"
OBS_LANGUAGE_TOKENS = "observation.language.tokens"
ACTION_IS_PAD = "action_is_pad"
