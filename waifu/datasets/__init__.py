import typing as t

T = t.TypeVar("T")


class BaseDataset(t.Generic[T]):
    '''Base dataset class.'''

    def __iter__(self):
        '''Implements the basic iterator interface.'''
        return self.generator()

    def generator(self) -> t.Generator[T, None, None]:
        '''Should yield individual items from the dataset.'''
        raise NotImplementedError
