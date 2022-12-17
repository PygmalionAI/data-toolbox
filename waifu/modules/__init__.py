import typing as t


class BaseModule:
    '''Base module class.'''

    def __iter__(self):
        '''Implements the basic iterator interface.'''
        return self.generator()

    def generator(self) -> t.Generator[str, None, None]:
        '''
        Should yield strings that will be used in the model's training /
        validation / test splits.
        '''
        raise NotImplementedError
