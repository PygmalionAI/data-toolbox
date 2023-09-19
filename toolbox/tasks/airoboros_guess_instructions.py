import logging
import random
import typing as t

from toolbox.core.models import Episode, Turn, TurnKind
from toolbox.core.task import BaseTask
from toolbox.datasets.airoboros import AiroborosDataset
from toolbox.utils.prompts import generate_prompts

LOG = logging.getLogger(__name__)

class AiroborosGuessTheInstructionTask(BaseTask):
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
                    utterance=instance.generation,
                    kind=TurnKind.USER,
                ),
                Turn(
                    utterance=instance.prompt,
                    kind=TurnKind.MODEL,
                ),
            ]

            yield Episode(turns=turns, identifier=f"airoboros-gti-{idx}")


_BASE_SYSTEM_PROMPTS = [
    "%{Enter|Engage|Begin|Consider} %{instruction guessing|reverse instruction} mode. In this mode, a user will type some %{text|answer|information} and %{the AI|you} will attempt to guess the instruction which %{corresponds|aligns with} the user's input. Do not say anything else but the instruction.",
    "%{Mode|Task}: 'Guess The Instruction'\nA user will type %{text|answer|information} and it is %{your|the AI's|the assistant's} %{job|goal} to answer with a generated instruction. Think of this almost like a question-guessing game.",
    "You are now in %{flipped instruction|reverse instruction|instruction guessing} mode. The %{user|prompter} will type something like an %{AI-|artificially }generated answer and you will provide the instruction that was used to %{generate|create} that answer.",
    "I am an %{assistant|AI} designed to %{guess|predict} what a user %{may|could|might} type as a question. The %{user|prompter} will send some sort of information and %{perhaps|maybe} some additional context in order for me to do so.",
    "Your question will be...",
    "%{I|I'll|i|i'll} %{predict|guess|foresee} whatever question you'll ask, given an answer!"
    "instruct",
    "assistant"
]

SYSTEM_PROMPTS = generate_prompts(_BASE_SYSTEM_PROMPTS)
