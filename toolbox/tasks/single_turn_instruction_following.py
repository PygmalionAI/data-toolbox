import logging
import random
import typing as t

from toolbox.core.models import Episode, Turn, TurnKind
from toolbox.core.task import BaseTask
from toolbox.datasets.gpt4llm import AlpacaLikeDataInstance #, Gpt4LlmDataset
from toolbox.datasets.gpteacher import GpTeacherDataset
from toolbox.utils.prompts import generate_prompts

LOG = logging.getLogger(__name__)


class SingleTurnInstructionFollowingTask(BaseTask):
    '''Instruction following task based on Alpaca-like data.'''

    def __iter__(self) -> t.Generator[Episode, None, None]:
        # for idx, instance in enumerate(Gpt4LlmDataset()):
        #     yield _data_instance_to_episode(instance, idx, "gpt-4-all")

        for idx, instance in enumerate(GpTeacherDataset()):
            try:
                yield _data_instance_to_episode(instance, idx, "gpteacher")
            except ValueError:
                pass


def _data_instance_to_episode(
    instance: AlpacaLikeDataInstance,
    idx: int,
    source: str,
) -> Episode:
    turns: list[Turn] = []

    # For some reason, some training examples have an input that's just a
    # chopped off segment of the instruction. Not great, so let's handle those
    # as no-input examples.
    bad_input = instance.input in instance.instruction

    if instance.input and not bad_input:
        # We have a separate input, so let's construct the prompt using
        # a separate system prompt for the instruction.
        turns = [
            Turn(
                utterance=instance.instruction,
                kind=TurnKind.SYSTEM,
            ),
            Turn(
                utterance=instance.input,
                kind=TurnKind.USER,
            ),
            Turn(
                utterance=instance.output,
                kind=TurnKind.MODEL,
            ),
        ]
    else:
        # No input, so basically just user prompt and response, so we'll
        # need to make a fake system prompt.
        turns = [
            Turn(
                utterance=random.choice(SYSTEM_PROMPTS),
                kind=TurnKind.SYSTEM,
            ),
            Turn(
                utterance=instance.instruction,
                kind=TurnKind.USER,
            ),
            Turn(
                utterance=instance.output,
                kind=TurnKind.MODEL,
            ),
        ]

    return Episode(turns=turns, identifier=f"{source}-{idx}")


_BASE_SYSTEM_PROMPTS = [
    "",
    "assistant",
    "%{You are now in|Engage|Start|Enter|Consider} %{instruction following|instruction|question answering|assistant|AI assistant} mode. %{Respond to the user|Follow the user's instructions} %{as well as you can|to the best of your abilities}.",
    "Q&A:\nQ: %{What mode am I in|What am I doing|Who am I}?\nA: You're in %{assistant|instruction following} mode.\nQ: What does that mean?\nA: You%{'ve gotta| must|should} %{take in|be given} a question or %{command|demand}, then you answer it and/or do what it says."
    "%{Purpose|Goal|Job}: Assistant\n%{Procedure|Objective|Methods of achieving your goal}: %{Answer the user's questions|Follow the instructions|Obey commands}",
    "%{I am|I'm} %{a helper for a user|a helpful assistant|engaged in what one might call 'instruction' mode}. Given %{queries|user queries}, I am to %{correctly|accurately} answer these things (at least, as best as I can).",
    "Instruction mode!",
    "u %{have|need} to answer whatever i ask and do whatever i say! do it now!!!"
]

SYSTEM_PROMPTS = generate_prompts(_BASE_SYSTEM_PROMPTS)
