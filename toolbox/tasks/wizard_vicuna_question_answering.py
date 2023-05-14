import random
import re
import typing as t

from toolbox.core.models import Episode, Turn, TurnKind
from toolbox.core.task import BaseTask
from toolbox.datasets.wizard_vicuna import (WizardVicunaConversation,
                                            WizardVicunaDataset)
from toolbox.utils.prompts import generate_prompts


class WizardVicunaQuestionAnsweringTask(BaseTask):
    '''Question answering based on WizardVicuna data.'''

    def __iter__(self) -> t.Generator[Episode, None, None]:
        for idx, conversation in enumerate(WizardVicunaDataset()):
            if not _conversation_passes_quality_check(conversation):
                continue

            # Apparently, a bunch of generations end with "{" according to some
            # users on HuggingFace. I haven't seen this myself yet, but just to
            # be safe let's fix that here.
            model_response = conversation.gpt_response
            if model_response[-1] == "{":
                model_response = model_response[:-1]

            turns: list[Turn] = [
                Turn(
                    utterance=random.choice(SYSTEM_PROMPTS),
                    kind=TurnKind.SYSTEM,
                ),
                Turn(
                    utterance=conversation.human_question,
                    kind=TurnKind.USER,
                ),
                Turn(
                    utterance=model_response,
                    kind=TurnKind.MODEL,
                ),
            ]

            yield Episode(
                turns=turns,
                identifier=f"wizard-vicuna-{conversation.id}-{idx}",
            )


def _conversation_passes_quality_check(
        conversation: WizardVicunaConversation) -> bool:
    '''Attempts to detect known-bad conversations.'''

    # Some entries were split incorrectly, so the question is broken off and
    # continues in the "response". This is fairly easy to detect by looking for
    # responses starting with lowercase letters or spaces.
    if re.match(r"[a-z]", conversation.gpt_response[0]) is not None:
        return False
    if conversation.gpt_response[0] == " ":
        return False

    return True


SYSTEM_PROMPTS = generate_prompts([
    "%{You are now in|Engage|Start|Enter} %{instruction following|instruction|question answering|assistant|AI assistant} mode. %{Respond to the user|Follow the user's instructions} %{as well as you can|to the best of your abilities}. {{response_length_str}}.",
    "{{response_length_str}}. %{Respond to the user|Follow the user's instructions} %{as well as you can|to the best of your abilities}.",
    "%{Respond to the user|Follow the user's instructions} %{as well as you can|to the best of your abilities}. {{response_length_str}}.",
])
