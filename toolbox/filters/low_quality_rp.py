import logging
import re

from ..core import BaseFilter, Episode, TurnKind

LOG = logging.getLogger("LowQualityRpFilter")

THRESHOLD = 0.5

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
PATTERNS: list[re.Pattern] = [
    FLOATING_QUOTATIONS_PATTERN,
    MUSHED_QUOTATION_MARKS_PATTERN,
    MUSHED_PARENTHESIS_PATTERN,
    LOWERCASE_I_PATTERN,
    LINKS_PATTERN,
]

class LowQualityRpFilter(BaseFilter):
    '''
    Attempts to find signs that an RP entry would be "low quality" via a few heuristics.
    '''
    def __init__(self) -> None:
        # Init method is required.
        pass

    def should_keep(self, episode: Episode) -> bool:
        low_quality_messages = 0
        for turn in episode.turns:
            message = turn.utterance
            # Ignore the system prompt on this one.
            if turn.kind != TurnKind.SYSTEM:
                # Go through RegEx patterns.
                for pattern in PATTERNS:
                    if pattern.search(message) is not None:
                        low_quality_messages += 1
                        break
                # If the message is empty, we count it as a low-quality message.
                if message.strip() == "":
                    low_quality_messages += 1
        
        # If the ratio of low-quality messages is greater than the threshold,
        # drop the episode.
        if low_quality_messages / len(episode.turns) >= THRESHOLD:
            LOG.info(f"Episode {episode.identifier} dropped due to signs of low-quality RP!")
            return False
        
        return True
