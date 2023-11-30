import hashlib
import logging

from toolbox.core.turns import Episode, TurnKind

from ..core import BaseFilter

LOG = logging.getLogger("ExactDedupFilter")

class ExactDedupFilter(BaseFilter):
    '''
    This filter will scan for any duplicate entries and drop them.
    We use a dict/hashmap so that we take advantage of O(1) look up rather than
    O(n) list scanning.
    
    Note that all filters are done on the task level, meaning that there may be
    duplicate entries between different datasets/tasks. You should do another
    pass of deduping across the entire dataset after it's processed if this is
    a concern.
    '''
    def __init__(self) -> None:
        # Init method is required.
        # This hashmap will be filled with hashes of chats and their IDs
        # (for logging)
        self.hashmap: dict[str, str] = {}

    def should_keep(self, episode: Episode) -> bool:
        # Remove any system prompts, since those will have random variations
        # on them.
        checkable_turns = [
            t.utterance for t in episode.turns if t.kind != TurnKind.SYSTEM
        ]
        # Hash the turns
        to_hash = "".join(checkable_turns)
        hash = hashlib.sha256(to_hash.encode()).hexdigest()
        # Use a try-except statement to attempt to access the hashmap.
        # If there's no entry corresponding to the hash in the hashmap,
        # this means that the episode is unique.
        try:
            duped_identifier = self.hashmap[hash]
            # If we get past the lookup, this is a duplicate. Filter out.
            LOG.info(f"Episode {episode.identifier} dropped due to being \
            a duplication of episode {duped_identifier}!")
            return False
        except KeyError:
            # Not a duplicate, but add it now to the hashmap.
            self.hashmap[hash] = episode.identifier
    
        return True
