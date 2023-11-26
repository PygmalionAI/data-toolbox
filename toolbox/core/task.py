from .filter import BaseFilter
from .training_example import TrainingExample

class BaseTask:
    '''Base task class. Relies on config fed into this task.'''
    def __init__(self, filters: list[BaseFilter], **kwargs) -> None:
        # We don't call BaseTask directly, but put __init__ here to account for
        # config parameters and task-specific filters.
        self.filters = filters

    def should_keep(self, example: TrainingExample) -> bool:
        '''
        Filtering on a task-specific level.
        '''
        for filter in self.filters:
            if not filter.should_keep(example):
                return False
        return True
