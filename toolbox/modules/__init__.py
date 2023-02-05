import typing as t

from toolbox.core.models import Episode


class BaseModule:
    '''Base module class.'''

    def __iter__(self) -> t.Generator[Episode, None, None]:
        '''Implements the basic iterator interface.'''
        return self.generator()

    def generator(self) -> t.Generator[Episode, None, None]:
        '''
        Should yield dialogue turns that will be used in the model's training /
        validation / test splits.
        '''
        raise NotImplementedError
