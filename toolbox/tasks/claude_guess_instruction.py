import logging
import random
import typing as t

from toolbox.core.models import Episode, Turn, TurnKind
from toolbox.core.task import BaseTask
from toolbox.datasets.claude_multiround import ClaudeInstructDataset
from toolbox.utils.prompts import generate_prompts

LOG = logging.getLogger(__name__)

class ClaudeGuessTheInstructionTask(BaseTask):
    '''
    Given an answer and possibly context, task the AI to generate a proper instruction or question for it.
    Heavily inspired by "Guess the Instruction! Flipped Learning Makes Language Models Stronger Zero-Shot Learners"
    Paper: https://arxiv.org/abs/2210.02969 | Github: https://github.com/seonghyeonye/Flipped-Learning/tree/master
    '''
    def __iter__(self) -> t.Generator[Episode, None, None]:
        for round in ClaudeInstructDataset():
            # We fetch only the first exchange in the multiround conversation for this task.
            # Human always goes first, but let's make sure that's the case...
            if round.conversation[0]["from"] != "human" or round.conversation[1]["from"] != "gpt":
                LOG.warning(f"Example {round.id} does not have the standard format, skipping...")

            user_prompt = round.conversation[0]["value"]
            output = round.conversation[1]["value"]

            # Now we check if either of these messages are blank.
            # If so, drop the example.
            if user_prompt == "" or output == "":
                LOG.warning(f"Skipping example {round.id}, unable to complete a full conversation")
                continue

            # Make the turns and yield the episode.
            turns: list[Turn] = [
                Turn(
                    utterance=random.choice(SYSTEM_PROMPTS),
                    kind=TurnKind.SYSTEM
                ),
                Turn(
                    utterance=output,
                    kind=TurnKind.USER
                ),
                Turn(
                    utterance=user_prompt,
                    kind=TurnKind.MODEL
                )
            ]

            yield Episode(
                turns=turns,
                identifier=f"claude-gti-{round.id}"
            )

_BASE_SYSTEM_PROMPTS = [
    "%{Enter|Engage|Begin|Consider} {instruction guessing|reverse instruction} mode. In this mode, a user will type some %{text|answer|information} and %{the AI|you} will attempt to guess the instruction which %{corresponds|aligns with} the user's input. Do not say anything else but the instruction.",
    "%{Mode|Task}: 'Guess The Instruction'\nA user will type %{text|answer|information} and it is %{your|the AI's|the assistant's} %{job|goal} to answer with a generated instruction. Think of this almost like a question-guessing game.",
    "You are now in %{flipped instruction|reverse instruction|instruction guessing} mode. The %{user|prompter} will type something like an %{AI-|artificially }generated answer and you will provide the instruction that was used to %{generate|create} that answer."
]

SYSTEM_PROMPTS = generate_prompts(_BASE_SYSTEM_PROMPTS)
