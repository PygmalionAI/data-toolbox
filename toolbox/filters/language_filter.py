import logging

import langdetect

from toolbox.core.filter_criteria import FilterCriteria
from toolbox.core.models import Episode

LOG = logging.getLogger(__name__)


# NOTE: This has a considerable performance impact. Keep in mind when deciding
# whether to use, and _where_ to include (before/after augmentations).
class LanguageFilter(FilterCriteria):
    '''Filters out non-English episodes.'''

    def __init__(self) -> None:
        super().__init__()

        # Enforce determinism in the language detection code.
        langdetect.DetectorFactory.seed = 0

    def keep(self, episode: Episode) -> bool:
        episode_text = "\n".join([turn.utterance for turn in episode.turns])
        detected_language = langdetect.detect(episode_text)

        if detected_language == "en":
            return True

        LOG.debug("Detected likely non-English episode (%s)", detected_language)
        return False
