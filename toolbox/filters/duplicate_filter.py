import hashlib
import logging

from toolbox.core.filter_criteria import FilterCriteria
from toolbox.core.models import Episode

LOG = logging.getLogger(__name__)


class DuplicateFilter(FilterCriteria):
    '''Filters out duplicate episodes.'''

    def __init__(self) -> None:
        super().__init__()

        self.seen_hashes: set[str] = set()

    def keep(self, episode: Episode) -> bool:
        serialized_episode = str(episode)

        episode_hash = _calculate_hash_for(serialized_episode)
        if episode_hash in self.seen_hashes:
            LOG.debug("Detected duplicate episode (SHA512 collision: %s)",
                      episode_hash)
            return False

        self.seen_hashes.add(episode_hash)
        return True


def _calculate_hash_for(text: str) -> str:
    return hashlib.sha512(text.encode("utf-8")).hexdigest()
