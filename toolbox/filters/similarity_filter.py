import logging
import typing as t

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from toolbox.core.filter_criteria import FilterCriteria
from toolbox.core.models import Episode

LOG = logging.getLogger(__name__)


class SimilarityFilter(FilterCriteria):
    '''Filters out episodes where bot messages are too similar to eachother.'''

    def __init__(self) -> None:
        super().__init__()

        # TODO(11b): Make this a tunable parameter.
        self.similarity_threshold = 0.75
        self.vectorizer = CountVectorizer()

    def keep(self, episode: Episode) -> bool:
        bot_messages = [
            turn.utterance for turn in episode.turns if not turn.human_speaker
        ]

        similarity_matrix = self._calculate_similarity_scores(bot_messages)
        average_similarity_for_episode = 0.0
        for score in similarity_matrix[0]:
            if score == 1:
                # Ignore comparisons against equal strings.
                continue
            average_similarity_for_episode += score
            average_similarity_for_episode /= 2

        if average_similarity_for_episode < self.similarity_threshold:
            return True

        LOG.debug("Detected episode with high message similarity (%s >= %s)",
                  average_similarity_for_episode, self.similarity_threshold)
        return False

    def _calculate_similarity_scores(self, strings: list[str]) -> t.Any:
        '''
        Calculates similarity scores between the given strings.

        This is a roundabout way to try and _possibly_ detect the post-1.1 CAI
        looping behavior so we can handle it during the data preprocessing. Open to
        suggestions on how to improve this.
        '''
        x = self.vectorizer.fit_transform(strings)
        arr = x.toarray()

        sims = cosine_similarity(arr)
        return sims
