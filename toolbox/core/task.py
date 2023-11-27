from typing import Optional

from filters import BaseFilter
from turns import Episode

class BaseTask:
    '''Base task class. Relies on config fed into this task.'''
    def __init__(
        self,
        filters: list[BaseFilter],
        custom_prompts: Optional[list[str]] = None,
        **kwargs
    ) -> None:
        # We don't call BaseTask directly, but put __init__ here to account for
        # config parameters and task-specific filters.
        self.filters = filters
        # "Custom prompts" in this case refers to the ability for users to
        # specify their own prompts in the prompt config.
        self.custom_prompts = None

    def should_keep(self, example: Episode) -> bool:
        '''
        Filtering on a task-specific level.
        '''
        for filter in self.filters:
            if not filter.should_keep(example):
                return False
        return True
