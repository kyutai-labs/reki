"""LeRobot plugin for reki: ``lerobot-eval --policy.path=<reki checkpoint>``.

LeRobot imports this distribution (``lerobot_policy_*``) at startup; the imports register the
``reki`` policy type and the ``reki`` env type (``--env.type=reki``).
"""

from lerobot_policy_reki.configuration_reki import RekiConfig
from lerobot_policy_reki.env_reki import RekiEnv
from lerobot_policy_reki.modeling_reki import RekiPolicy
from lerobot_policy_reki.processor_reki import make_reki_pre_post_processors

__all__ = [
    "RekiConfig",
    "RekiEnv",
    "RekiPolicy",
    "make_reki_pre_post_processors",
]
