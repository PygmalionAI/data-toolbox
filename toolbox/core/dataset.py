from typing import Generator, Generic, TypeVar

T = TypeVar('T')

class BaseDataset(Generic[T]):
    '''The base dataset class which all unique dataset gatherers descend from.'''
    def __iter__(self) -> Generator[T, None, None]:
        '''
        This method must be overridden when inheriting. It should yield
        individual items from the dataset.
        '''
        raise NotImplementedError
