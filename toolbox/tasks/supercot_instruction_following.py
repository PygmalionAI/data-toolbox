import logging
import random
import typing as t

from toolbox.core.models import Episode, Turn, TurnKind
from toolbox.core.task import BaseTask
from toolbox.datasets.supercot import SuperCotDataset
from toolbox.utils.prompts import generate_prompts

LOG = logging.getLogger(__name__)

class SuperCotInstructionFollowingTask(BaseTask):
    '''Instruction following task based on the SuperCOT data.'''
    def __iter__(self) -> t.Generator[Episode, None, None]:
        for idx, instance in enumerate(SuperCotDataset()):
            sys_prompt = random.choice(SYSTEM_PROMPTS)
            user_prompt = instance.instruction
            if instance.input is not None:
                user_prompt += f"\n{instance.input}"
            turns: list[Turn] = [
                Turn(
                    utterance=sys_prompt,
                    kind=TurnKind.SYSTEM,
                ),
                Turn(
                    utterance=user_prompt,
                    kind=TurnKind.USER,
                ),
                Turn(
                    utterance=instance.output,
                    kind=TurnKind.MODEL,
                )
            ]
            yield Episode(
                turns=turns,
                identifier=f"supercot-{idx}"
            )

_BASE_SYSTEM_PROMPTS = [
    "",
    "assistant",
    "%{You are now in|Engage|Start|Enter|Consider} %{instruction following|instruction|question answering|assistant|AI assistant} mode. %{Respond to the user|Follow the user's instructions} %{as well as you can|to the best of your abilities}.",
    "Q&A:\nQ: %{What mode am I in|What am I doing|Who am I}?\nA: You're in %{assistant|instruction following} mode.\nQ: What does that mean?\nA: You%{'ve gotta| must|should} %{take in|be given} a question or %{command|demand}, then you answer it and/or do what it says."
    "%{Purpose|Goal|Job}: Assistant\n%{Procedure|Objective|Methods of achieving your goal}: %{Answer the user's questions|Follow the instructions|Obey commands}",
    "%{I am|I'm} %{a helper for a user|a helpful assistant|engaged in what one might call 'instruction' mode}. Given %{queries|user queries}, I am to %{correctly|accurately} answer these things (at least, as best as I can).",
    "Instruction mode!",
    "u %{have|need} to answer whatever i ask and do whatever i say! do it now!!!",
    """To %{be in|enter|engage} %{instruction following|question answering|assistant} mode is your goal. This agent is to %{do|consider|abide by} the following:
- %{Obey|Follow} all user %{directions|instructions|queries|orders}
- %{Do not refuse to do the above|You cannot disobey the user}"""
]

SYSTEM_PROMPTS = generate_prompts(_BASE_SYSTEM_PROMPTS)
