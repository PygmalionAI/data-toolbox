from typing import Type

from ..core import BaseTask
from .airoboros_instruction_following import AiroborosInstructionFollowingTask
from .characterai_roleplay import CharacterAiRoleplayTask
from .rp_forums_roleplay import RpForumsRoleplayTask
from .teatime_roleplay import TeatimeRoleplayTask

# Make this more dynamic later.
NAME_TO_TASK_MAPPING: dict[str, Type[BaseTask]] = {
    cls.__name__: cls for cls in [
        AiroborosInstructionFollowingTask,
        CharacterAiRoleplayTask,
        RpForumsRoleplayTask,
        TeatimeRoleplayTask
    ]
}
