import logging

from toolbox.core.turns import Episode

from ..core import BaseFilter

LOG = logging.getLogger("AppleFilter")

class AppleFilter(BaseFilter):
    '''
    A test filter. It simply detects if any message contains the substring
    of "apple", and if so, drops the example. Obviously this shouldn't be used
    for any serious dataset, unless you really really hate apples.
    '''
    @staticmethod
    def should_keep(episode: Episode) -> bool:
        for i, turn in enumerate(episode.turns):
            if "apple" in turn.utterance:
                LOG.info(f"Episode {episode.identifier} dropped due to it \
                containing substring 'apple' in turn {i+1}!")
                return False
        return True
