import typing as t

from toolbox.core.models import Episode


class BaseTask:
    '''Base task class.'''

    def __iter__(self) -> t.Generator[Episode, None, None]:
        '''This method must be overidden when inheriting.'''
        raise NotImplementedError
