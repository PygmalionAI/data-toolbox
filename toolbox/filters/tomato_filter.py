import logging

from toolbox.core.filter_criteria import FilterCriteria
from toolbox.core.models import Episode

LOG = logging.getLogger(__name__)


class TomatoFilter(FilterCriteria):
    '''
    Filters out episodes where the bot blushes and/or stutters excessively.
    '''

    def __init__(self) -> None:
        # TODO(11b): Make these into tunable parameters.
        self.embarassment_threshold = 0.7
        self.stuttering_threshold = 0.7

        super().__init__()

    def keep(self, episode: Episode) -> bool:
        bot_messages = [
            turn.utterance for turn in episode.turns if not turn.human_speaker
        ]

        stutter_ratio = 0.0
        for message in bot_messages:
            stutter_ratio += 1.0 if _has_stuttering(message) else 0.0
            stutter_ratio /= 2
        if stutter_ratio >= self.stuttering_threshold:
            LOG.debug("Detected excessive stuttering (%s >= %s)", stutter_ratio,
                      self.stuttering_threshold)
            return False

        embarassment_ratio = 0.0
        for message in bot_messages:
            embarassment_ratio += 1.0 if _seems_embarassed(message) else 0.0
            embarassment_ratio /= 2
        if embarassment_ratio >= self.embarassment_threshold:
            LOG.debug("Detected excessive embarassment (%s >= %s)",
                      embarassment_ratio, self.embarassment_threshold)
            return False

        return True


def _seems_embarassed(utterance: str) -> bool:
    '''Most advanced sentiment classification algorithm ever invented.'''
    last_idx = -1

    # Only look within asterisks (e.g. *text like this*).
    asterisk_idxs = []
    while (last_idx := utterance.find("*", last_idx + 1)) != -1:
        asterisk_idxs.append(last_idx)

    while True:
        if len(asterisk_idxs) < 2:
            break

        start_idx = asterisk_idxs.pop(0)
        end_idx = asterisk_idxs.pop(0)
        action = utterance[start_idx:end_idx].lower()

        if any([
                x in action for x in
            ["blush", "like a tomato", "as a tomato", "embarass", "nervous"]
        ]):
            return True
    return False


def _has_stuttering(utterance: str) -> bool:
    '''Tries to detect stuttering like "W-what?"'''
    for word in utterance.split():
        if "-" not in word:
            continue

        if len(word) < 4:
            continue

        if word[1] != "-":
            continue

        lowercased_word = word.lower()
        if lowercased_word[0] == lowercased_word[2]:
            return True

    return False
