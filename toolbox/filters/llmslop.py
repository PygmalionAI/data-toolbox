import logging

from ..core import BaseFilter, Episode, TurnKind

LOG = logging.getLogger("LlmSlopFilter")

# https://github.com/AlpinDale/gptslop/blob/main/gptslop.yaml
GPT_SLOP = [
    "as an ai",
    "i'm sorry, but",
    "however, it is important to note",
    "in conclusion,"
]

# NOTE(TG): There's a lot of common phrases in here. Remove some?
# https://github.com/AlpinDale/gptslop/blob/main/claudeslop.yaml
CLAUDE_SLOP = [
    "ministrations",
    "audible pop",
    "rivulets of",
    "admit it",
    "the ball is in your court",
    "the game is on",
    "the choice is yours",
    "i don't bite... unless you want me to",
    "half-lidded_eyes",
    "she worries her bottom lip",
    "warring with",
    "arousal pooling in her belly",
    "take your pleasure",
    "fiddles with the hem of her skirt",
    "kiss-bruised lips",
    "a bruising kiss",
    "despite herself",
    "yours to take",
    "wanton",
    "with reckless abandon",
    "torn between",
    "knuckles turning white",
    "grins wickedly",
    "fiery red hair",
    "long lashes",
    "propriety be damned",
    "the world narrows",
    "pupils blown wide with pleasure",
    "tongue darts out",
    "chestnut eyes",
    "grasps your chin and forces you to meet her gaze",
    "bites your ear",
    "nails raking angry red lines down your back",
    "her cheeks flaming",
    "cheeks hollowing",
    "stars burst behind her eyes",
    "inner walls clenching around nothing",
    "puckered hole",
    "her wet heat",
    "she whimpers, biting her lip",
    "dusky nipples",
    "slick folds",
    "still lodged deep inside her",
    "heart, body and soul belong to you",
    "the night is still young",
    "...for now.",
    "whether you like it or not",
    "without waiting for a response",
    "claude"
]

SLOP = GPT_SLOP + CLAUDE_SLOP

class LlmSlopFilter(BaseFilter):
    '''
    This filter is designed to try to remove the "voice" of certain popular
    LLMs such as GPT-3.5/4 and Claude by dropping any "GPTisms" and "Claudisms"
    which influence the voice of the model being trained on this data.
    We keep it separate from the RefusalFilter for maximum flexibility with
    whatever the user wants to filter out of their data.
    '''
    def __init__(self) -> None:
        # Init method is required.
        pass

    def should_keep(self, episode: Episode) -> bool:
        for i, turn in enumerate(episode.turns, start=1):
            message = turn.utterance
            if turn.kind != TurnKind.SYSTEM:
                # Go through the slop
                for phrase in SLOP:
                    if phrase in message.lower():
                        LOG.debug(f"Episode {episode.identifier} dropped due to detected slop '{phrase}' in turn {i}!")
                        return False
        return True
