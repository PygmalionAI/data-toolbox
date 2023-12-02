from typing import Type

from ..core import BaseTask
from .airoboros_instruction_following import AiroborosInstructionFollowingTask
from .rp_forums_roleplay import RpForumsRoleplayTask

# Make this more dynamic later.
NAME_TO_TASK_MAPPING: dict[str, Type[BaseTask]] = {
    cls.__name__: cls for cls in [
        AiroborosInstructionFollowingTask,
        RpForumsRoleplayTask
    ]
}
