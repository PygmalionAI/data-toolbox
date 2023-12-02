import logging

from ..core import BaseFilter, Episode

LOG = logging.getLogger("AppleFilter")

class AppleFilter(BaseFilter):
    '''
    A test filter. It simply detects if any message contains the substring
    of "apple", and if so, drops the example. Obviously this shouldn't be used
    for any serious dataset, unless you really really hate apples.
    '''
    def __init__(self) -> None:
        # Init method is required.
        pass

    def should_keep(self, episode: Episode) -> bool:
        for i, turn in enumerate(episode.turns, start=1):
            if "apple" in turn.utterance:
                LOG.info(f"Episode {episode.identifier} dropped due to it containing substring 'apple' in turn {i}!")
                return False
        return True
