import logging
import random
import typing as t

from toolbox.core.models import Episode, Turn, TurnKind
from toolbox.core.task import BaseTask
from toolbox.datasets.airoboros import AiroborosDataset
from toolbox.utils.prompts import generate_prompts

LOG = logging.getLogger(__name__)

class AiroborosInstructionFollowingTask(BaseTask):
    '''Instruction following task based on the Airoboros data.'''
    def __iter__(self) -> t.Generator[Episode, None, None]:
        for idx, instance in enumerate(AiroborosDataset(), start=1):
            # Throw out any responses containing "Airoboros"
            if instance.generation.lower().strip() == "airoboros":
                continue

            turns: list[Turn] = [
                Turn(
                    utterance=random.choice(SYSTEM_PROMPTS),
                    kind=TurnKind.SYSTEM,
                ),
                Turn(
                    utterance=instance.prompt,
                    kind=TurnKind.USER,
                ),
                Turn(
                    utterance=instance.generation,
                    kind=TurnKind.MODEL,
                ),
            ]

            yield Episode(turns=turns, identifier=f"airoboros-instruct-{idx}")


BASE_SYSTEM_PROMPTS = [
    "",
    "%{Enter|Engage|Begin in} assistant mode. Answer the user's questions in a detailed manner.",
    "%{You are now in|Engage|Start|Enter} %{instruction following|instruction|question answering|assistant|AI assistant} mode. %{Respond to the user|Follow the user's instructions} %{as well as you can|to the best of your abilities}.",
    "%{Respond to the user|Follow the user's instructions} %{as well as you can|to the best of your abilities}.",
    "%{Purpose|Goal|Job}: Assistant\n%{Procedure|Objective|Methods of achieving your goal}: %{Answer the user's questions|Follow the user's instructions}"
]

SYSTEM_PROMPTS = generate_prompts(BASE_SYSTEM_PROMPTS)
