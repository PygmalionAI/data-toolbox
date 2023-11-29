from abc import ABC, abstractmethod
from .turns import Episode

class BaseFormat(ABC):
    '''
    Base format from which all formats inherit from.
    The format is what will turn examples into dictionaries which will be
    dumped into the output JSONL file.
    '''
    def __init__(self) -> None:
        # still need to figure out exactly how to do this
        raise NotImplementedError
    
    @abstractmethod
    def apply_format(self, episode: Episode) -> Episode:
        raise NotImplementedError
    
    @abstractmethod
    def construct_dict(self, episode: Episode) -> dict:
        raise NotImplementedError
