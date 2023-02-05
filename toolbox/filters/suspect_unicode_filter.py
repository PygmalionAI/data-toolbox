import logging

import regex

from toolbox.core.filter_criteria import FilterCriteria
from toolbox.core.models import Episode

LOG = logging.getLogger(__name__)


class SuspectUnicodeFilter(FilterCriteria):
    '''Filters out episodes that contain weird characters.'''

    def __init__(self) -> None:
        super().__init__()

        self.seen_hashes: set[str] = set()

    def keep(self, episode: Episode) -> bool:
        episode_text = "\n".join([turn.utterance for turn in episode.turns])

        if _contains_suspect_unicode(episode_text):
            LOG.debug("Detected suspect Unicode codepoints (`%s`)",
                      episode_text)
            return False

        return True


def _contains_suspect_unicode(string: str) -> bool:
    '''
    Returns whether the given string seems to have suspect Unicode trickery
    (e.g.: Zalgo text).
    '''
    return regex.search(r"\pM{3,}", string) is not None
