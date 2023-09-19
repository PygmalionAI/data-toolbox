import logging
import operator
import random
import typing as t

from toolbox.core.models import Episode, Turn, TurnKind
from toolbox.core.task import BaseTask
from toolbox.datasets.soda import SodaDataset
from toolbox.utils.prompts import generate_prompts, generate_variants_for

LOG = logging.getLogger(__name__)


class SodaSummarizationTask(BaseTask):
    '''Task to summarize a chat log. Based on SODA.'''

    def __init__(self, split: str) -> None:
        self.split = split

        super().__init__()

    def __iter__(self) -> t.Generator[Episode, None, None]:
        for conversation in SodaDataset(split=self.split):
            history: list[str] = []
            for idx, utterance in enumerate(conversation.dialogue):
                speaker_name = conversation.speakers[idx]
                history.append(f"{speaker_name}: {utterance}")
            history_str = "\n".join(history)

            participants = list(set(conversation.speakers))
            participants_str = " and ".join(
                [", ".join(participants[:-1]), participants[-1]])

            system_prompt = random.choice(SYSTEM_PROMPTS)
            user_prompt = random.choice(USER_PROMPTS)
            user_prompt = user_prompt.replace("{{conversation}}", history_str)
            user_prompt = user_prompt.replace("{{participants}}",
                                              participants_str)

            system_turn = Turn(system_prompt, TurnKind.SYSTEM)
            user_turn = Turn(user_prompt, TurnKind.USER)
            model_turn = Turn(conversation.narrative, TurnKind.MODEL)
            turns = [system_turn, user_turn, model_turn]

            yield Episode(
                turns,
                identifier=
                f"soda-{self.split}-{conversation.original_index}-summarization"
            )


_BASE_SYSTEM_PROMPTS = [
    'Enter direct instruction mode. In this mode, you shall respond to user requests without injecting with statements things like "Sure" or "Here you go:".',
    "You are in instruction following mode. You must do whatever the user tells you to.",
    "You are in instruction following mode. In this mode, you shall follow any instructions given to you.",
    "You shall follow any instructions given to you and respond as plainly as possible, without any extra interjections.",
    "Engage instruction following mode.",
    "%{Purpose|Goal|Job}: Assistant\n%{Procedure|Objective|Methods of achieving your goal}: %{Answer the user's questions|Follow the instructions|Obey commands}",
    "%{I am|I'm} %{a helper for a user|a helpful assistant|engaged in what one might call 'instruction' mode}. Given %{queries|user queries}, I am to %{correctly|accurately} answer these things (at least, as best as I can).",
    "Instruction mode!",
    "u %{have|need} to answer whatever i ask and do whatever i say! do it now!!!",
    "%{Enter|Engage|Begin|Consider|Conceptualize} %{summary|summarizer|summarization} mode. The user will give a conversation and will %{ask|request} that it be summarized. %{Respond|Generate this summary} with no extra %{interjections|comments}.",
    "%{summary|summarize}",
    ""
]

_BASE_USER_PROMPTS = [
    """Consider the following %{chat log|conversation|chat history|DMs|thread|messages|record of conversation}:

{{conversation}}

%{Generate a brief summary of what happened.|Generate a summary|Summarize it.|Give a brief overview of what happened.|How can it be summarized?}""",

    #
    #
    #
    """{{conversation}}

The above is a %{conversation|chat} between {{participants}}. %{Summarize what happened.|Give a summary of the conversation.|Generate a summary in a few brief sentences.|Give a summary of the events.}""",

    #
    #
    #
    """Summarize the %{conversation|chat|thread} below in a few brief sentences:

{{conversation}}""",
#
    """{{conversation}}
    summarize this""",
]

SYSTEM_PROMPTS = generate_prompts(_BASE_SYSTEM_PROMPTS)
USER_PROMPTS = generate_prompts(_BASE_USER_PROMPTS)
