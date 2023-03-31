import logging
import random
import re
import typing as t

from toolbox.core.models import Episode, Turn, TurnKind
from toolbox.core.task import BaseTask
from toolbox.datasets.rp_data import RPDataset
from toolbox.utils.prompts import generate_prompts

LOG = logging.getLogger(__name__)

class HumanRoleplayTask(BaseTask):
    '''Task to generate an appropriate roleplay response given the last response(s).'''
    def __init__(self, keep_ooc: bool = False) -> None:
        # OOC might provide a certain "charm" to the bot which
        # we might want to keep.
        self.keep_ooc = keep_ooc
        super().__init__()

    def __iter__(self) -> t.Generator[Episode, None, None]:
        for thread in RPDataset():
            # If thread is only 1 message long, cut it out
            if len(thread.messages) <= 1:
                LOG.debug(f'Skipping thread "{thread.thread_name}" with only one message')
                continue

            # System prompt
            system_prompt = random.choice(SYSTEM_PROMPTS)
            system_turn = Turn(utterance=system_prompt, kind=TurnKind.SYSTEM)
            turns: list[Turn] = [system_turn]

            # TODO(TG): Find a better way to gather authors.
            # This implementation only accounts for 2 authors doing back and forth
            for i, message in enumerate(thread.messages):
                cleaned_message = message.message
                if not self.keep_ooc:
                    cleaned_message = OOC_REGEX.sub('', cleaned_message).strip()

                turn = Turn(
                    utterance=cleaned_message,
                    kind=TurnKind.USER if i % 2 == 0 else TurnKind.MODEL
                )
                turns.append(turn)

            yield Episode(turns=turns, identifier=f"rp-{thread.thread_name}")

OOC_REGEX = re.compile(r"\((\(|(OOC)).*?\)?\)")

# TODO(TG): Implement a function in to fully deal with HTML in the data.
# We want to convert that to Markdown when applicable,
# and remove it when not applicable.

_BASE_SYSTEM_PROMPTS = [
    """Enter %{roleplaying|roleplay|RP} mode. Your %{objective|goal|job} is to roleplay with the user given %{a history of responses|context}. %{Create|Make} long and interesting replies to the user's input, and stay on topic.""",
    """You are now in %{roleplay|roleplay conversational} mode. You will generate long, interesting, and detailed %{dialog|dialogue|responses} to the user given a contextual history.""",
    # Experimental idea: treat this as a text sequence generation task,
    # rather than a strictly roleplaying task in order to increase
    # flexibility in prompts/purpose
    """You are a %{roleplay|RP|conversational} model whose %{task|job|objective} is to generate the next %{reply|part|element} in a sequence of text. The generated text will be long, detailed and %{on-topic|relevant to the previous entries of the sequence}."""
]

SYSTEM_PROMPTS = generate_prompts(_BASE_SYSTEM_PROMPTS)