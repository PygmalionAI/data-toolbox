import logging
import random
import typing as t

from toolbox.core.models import Episode, Turn, TurnKind
from toolbox.core.task import BaseTask
from toolbox.datasets.oasst import OpenAssistantDataset

LOG = logging.getLogger(__name__)

class OpenAssistantInstructionFollowingTask(BaseTask):
    '''
    Multi-turn instruction following from OpenAssistant's dataset.

    Params:
    kept_languages: The languages to keep in the OpenAssistant dataset. Set to None to keep all languages.
    '''
    def __init__(self, kept_languages: t.Optional[list[str]] = None) -> None:
        self.kept_languages = kept_languages
        super().__init__()

    def __iter__(self) -> t.Generator[Episode, None, None]:
        # Cache parent tree id for identifier string
        tree_id = ""
        for conversation in OpenAssistantDataset():
            # If the language of the conversation is not in the kept languages, skip it
            if self.kept_languages is not None and conversation.language not in self.kept_languages:
                continue

            # Set tree id and counter for identifier string
            if conversation.tree_id != tree_id:
                tree_id = conversation
                counter = 1

            turns: list[Turn] = [
                Turn(
                    utterance=random.choice(SYSTEM_PROMPTS),
                    kind=TurnKind.SYSTEM
                )
            ]

            # Create turns by iterating through conversation. Messy but should work
            turns += [Turn(
                utterance=c["text"],
                kind=TurnKind.USER if c["role"] == "prompter" else TurnKind.MODEL
            ) for c in conversation]

            yield Episode(turns=turns, identifier=f"oasst-{tree_id}-{counter}")
            counter += 1


SYSTEM_PROMPTS = [
    "Consider Assistant, a large language model (LLM). It responds to user requests as truthfully as it can.",
    "You are a large language model trained to act as an assistant. You are to follow user instructions and answer user questions to the best of your abilities.",
    "Enter assistant mode. In this mode, you will follow instructions and respond with helpful responses.",
    "You are now in assistant mode. You shall follow user instructions and answer user questions by responding with helpful, actionable messages.",
    "Assistant, engage instruction following and question answering mode. You are bound to generating text, and cannot perform any other actions.",
    "Consider Assistant, a LLM trained to follow user instructions and answer questions.",
]