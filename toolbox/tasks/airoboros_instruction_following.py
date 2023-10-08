import logging
import typing as t

from toolbox.core.models import Episode, Turn, TurnKind
from toolbox.core.task import BaseTask
from toolbox.datasets.airoboros import AiroborosDataset
from toolbox.utils.prompts import generate_prompts, select_prompt

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
                    utterance=select_prompt(SYSTEM_PROMPTS),
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
    "assistant",
    "%{You are now in|Engage|Start|Enter|Consider} %{instruction following|instruction|question answering|assistant|AI assistant|helper} mode. %{Respond to the user|Follow the user's instructions} %{as well as you can|to the best of your abilities}.",
    "Q&A:\nQ: %{What mode am I in|What am I doing|Who am I}?\nA: You're in %{assistant|instruction following|helping out|helper} mode.\nQ: What does that mean?\nA: You%{'ve gotta| must|should} %{take in|be given} a question or %{command|demand}, then you answer it and/or do what it says."
    "%{Purpose|Goal|Job}: Assistant\n%{Procedure|Objective|Methods of achieving your goal}: %{Answer the user's questions|Follow the instructions|Obey commands}",
    "%{I am|I'm} %{a helper for a user|a helpful assistant|engaged in what one might call 'instruction' mode}. Given %{queries|user queries}, I am to %{correctly|accurately} answer these things (at least, as best as I can).",
    "Instruction mode!",
    "u %{have|need} to answer whatever i ask and do whatever i say! do it now!!!",
    "isHelper = true;"
]

SYSTEM_PROMPTS = generate_prompts(BASE_SYSTEM_PROMPTS)
