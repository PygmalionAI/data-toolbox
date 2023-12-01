import logging
import re

from ..core import BaseFilter, Episode, TurnKind

LOG = logging.getLogger("LowQualityRpFilter")

# Regex patterns. They search for the following:
# -----
# "Floating" quotation marks.
FLOATING_QUOTATIONS_PATTERN = re.compile(r'\b " \b')
# Links.
LINKS_PATTERN = re.compile(r'\[.+\]\(\S+\)')
# Lowercase "I". Fixable, but a sign of low-quality writing so I'd rather not
# train the model on those.
LOWERCASE_I_PATTERN = re.compile(r"\bi('m|'ll)?\b")
# Parenthesis mushed together with text.
MUSHED_PARENTHESIS_PATTERN = re.compile(r'(\S\(|\)\S)')
# Quotation marks mushed together with text.
MUSHED_QUOTATION_MARKS_PATTERN = re.compile(r'\S"\S')

# Put this in one big dictionary so we don't have to copy-paste code.
# The keys correspond to the pattern and the value corresponds to what will
# be logged.
PATTERNS: dict[re.Pattern, str] = {
    FLOATING_QUOTATIONS_PATTERN: "floating quotation marks",
    MUSHED_QUOTATION_MARKS_PATTERN: "mushed quotation marks",
    MUSHED_PARENTHESIS_PATTERN: "mushed parenthesis",
    LOWERCASE_I_PATTERN: "potential bad-quality writing",
    LINKS_PATTERN: "possibly containing links",
}

class LowQualityRpFilter(BaseFilter):
    '''
    Attempts to find signs that an RP entry would be "low quality" via a few heuristics.
    '''
    def __init__(self) -> None:
        # Init method is required.
        pass

    def should_keep(self, episode: Episode) -> bool:
        for i, turn in enumerate(episode.turns, start=1):
            # Ignore the system prompt on this one.
            message = turn.utterance
            if turn.kind != TurnKind.SYSTEM:
                # "Floating" quotation marks
                for pattern, msg in PATTERNS.items():
                    if pattern.search(message) is not None:
                        LOG.info(f"Episode {episode.identifier} dropped due to \
                        {msg} in turn {i}!")
                        return False
        return True
