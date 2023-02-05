import abc

from toolbox.core.models import Episode


class FilterCriteria(abc.ABC):
    '''
    Abstract class. Defines an interface for filters that operate upon
    individual episodes.
    '''

    @abc.abstractmethod
    def keep(self, episode: Episode) -> bool:
        '''
        Should return whether or not the given `episode` should be kept in the
        dataset.
        '''
        raise NotImplementedError
