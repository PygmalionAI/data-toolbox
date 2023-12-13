import logging

from typing import Any

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..core import BaseFilter, Episode, TurnKind

LOG = logging.getLogger("SimilarityFilter")

class SimilarityFilter(BaseFilter):
    '''
    This filter attempts to weed out episodes where the model has repetitively
    generated similar responses. If the similarity is above a certain threshold, drop it.
    '''
    def __init__(self) -> None:
        # Init method is required.
        self.threshold = 0.6 # TODO(TG): Make this configurable.
        self.vectorizer = CountVectorizer()

    def _calculate_similarity_scores(self, strings: list[str]) -> Any:
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

    def should_keep(self, episode: Episode) -> bool:
        # Don't bother caring about episodes with only one exchange.
        if len(episode.turns) <= 3:
            return True
        
        bot_messages = [turn.utterance for turn in episode.turns if turn.kind == TurnKind.MODEL]
        try:
            similarity_matrix = self._calculate_similarity_scores(bot_messages)
        except ValueError:
            # A message is likely comprised of only stopwords, so sklearn
            # blows up. Assume best-case scenario and assume messages are not similar
            # to each other.
            similarity_matrix = [[0]]

        avg_similarity_for_episode = 0.
        for score in similarity_matrix[0]:
            if score == 1:
                # Ignore comparisons against equal strings.
                continue
            avg_similarity_for_episode += score
            avg_similarity_for_episode /= 2

        if avg_similarity_for_episode >= self.threshold:
            LOG.debug(f"Episode {episode.identifier} dropped due to similarity score of {avg_similarity_for_episode}!")
            return False
        
        return True
