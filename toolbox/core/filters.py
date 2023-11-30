from abc import ABC, abstractmethod
from .turns import Episode

class BaseFilter(ABC):
    '''
    Any filter for data should inherit from this base class.
    Filters work on the task level and discards any data that does not meet
    a certain criteria (must be English, must not be a duplicate, etc etc.)
    '''
    @staticmethod    
    @abstractmethod
    def should_keep(episode: Episode) -> bool:
        '''
        Whether or not the given training episode should be kept and used
        for training.
        This class should use a logger for the purposes of informing the user
        that a specific example did not pass the specific filter.
        '''
        raise NotImplementedError
