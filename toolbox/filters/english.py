import logging

from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

from ..core import BaseFilter, Episode

LOG = logging.getLogger("EnglishFilter")

DETECTOR_THRESHOLD = 0.6
# Set up a seed for consistent results.
DetectorFactory.seed = 42

class EnglishFilter(BaseFilter):
    '''
    This filter uses langdetect to weed out any non-English conversations.
    A threshold is specified. If the ratio of non-English messages to total messages
    '''
    def __init__(self) -> None:
        # Init method is required.
        pass

    def should_keep(self, episode: Episode) -> bool:
        num_non_english = 0
        # Ignore system prompt, which is more than likely always English.
        for turn in episode.turns[1:]:
            try:
                if detect(turn.utterance) != "en":
                    num_non_english += 1
            except LangDetectException:
                # If the language detection fails, we assume that it's not English.
                num_non_english += 1
        
        if num_non_english / len(episode.turns) >= DETECTOR_THRESHOLD:
            LOG.info(f"Episode {episode.identifier} dropped due to majority of conversation not being in English!")
            return False

        return True
