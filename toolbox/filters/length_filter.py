import logging
import random
import statistics

from toolbox.core.filter_criteria import FilterCriteria
from toolbox.core.models import Episode

LOG = logging.getLogger(__name__)


class LengthFilter(FilterCriteria):
    '''Randomly drops episodes with short responses.'''

    def __init__(self) -> None:
        # Anything above or at this threshold will always be kept.
        self.DESIRED_MEDIAN_WORD_COUNT = 48

        # Anything below or at this threshold will have a 100% chance of being
        # dropped.
        self.MINIMUM_MEDIAN_WORD_COUNT = 2

        super().__init__()

    def keep(self, episode: Episode) -> bool:
        word_counts = [
            # Build an array containing the word count...
            len(turn.utterance.split())
            # ...of every non-human turn.
            for turn in episode.turns
            if not turn.human_speaker
        ]

        median_word_count = statistics.median(word_counts)
        random_float = random.random()
        chance_to_drop = self._chance_to_drop(median_word_count)

        should_drop = chance_to_drop <= random_float
        return should_drop

    def _chance_to_drop(self, median_word_count: int) -> float:
        '''
        Given a median word count for an episode, returns the % chance of it
        being dropped.
        '''
        interp_points = (
            (self.MINIMUM_MEDIAN_WORD_COUNT, 1.0),
            (self.DESIRED_MEDIAN_WORD_COUNT, 0.0),
        )

        unclamped_drop_chance = _interpolate(interp_points, median_word_count)
        return _clamp(0.0, 1.0, unclamped_drop_chance)


def _interpolate(data: list[list[int]], x: int) -> float:
    '''Linear interpolation implementation.'''
    return data[0][1] + (x - data[0][0]) * ((data[1][1] - data[0][1]) /
                                            (data[1][0] - data[0][0]))


def _clamp(min_value: float, max_value: float, value_to_clamp: float) -> float:
    '''Clamps `value_to_clamp` so `min_value <= value_to_clamp <= max_value`.'''
    return sorted((min_value, value_to_clamp, max_value))[1]
