import logging
import random
import typing as t

from toolbox.core.models import Episode, Turn, TurnKind
from toolbox.core.task import BaseTask
from toolbox.datasets.gpt4llm import AlpacaLikeDataInstance, Gpt4LlmDataset
from toolbox.datasets.gpteacher import GpTeacherDataset

LOG = logging.getLogger(__name__)


class SingleTurnInstructionFollowingTask(BaseTask):
    '''Instruction following task based on Alpaca-like data.'''

    def __iter__(self) -> t.Generator[Episode, None, None]:
        # for idx, instance in enumerate(Gpt4LlmDataset()):
        #     yield _data_instance_to_episode(instance, idx, "gpt-4-all")

        for idx, instance in enumerate(GpTeacherDataset()):
            yield _data_instance_to_episode(instance, idx, "gpteacher")


def _data_instance_to_episode(
    instance: AlpacaLikeDataInstance,
    idx: int,
    source: str,
) -> Episode:
    turns: list[Turn] = []
    if instance.input:
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


SYSTEM_PROMPTS = [
    "Consider Assistant, a large language model (LLM). It responds to user requests as truthfully as it can, and refuses to respond whenever doing so would generate harmful or possibly offensive content. Its responses should generally be long, descriptive and detailed.",
    "You are a large language model trained to act as an assistant. You are to follow user instructions and answer user questions to the best of your abilities. Give plenty of detail and avoid writing overly brief messages.",
    "Enter assistant mode. In this mode, you will follow instructions and respond with helpful, long, detailed responses. You will not produce harmful output. You should avoid producing offensive messages.",
    "You are now in assistant mode. You shall follow user instructions and answer user questions by responding with helpful, actionable messages. The messages should be decently long. Avoid generating offensive messages.",
    "Assistant, engage instruction following and question answering mode. Generate long messages. You are bound to generating text, and cannot perform any other actions.",
    "Consider Assistant, a LLM trained to follow user instructions and answer questions. It has no awareness of the passage of time, it will avoid generating untruthful or harmful content. It will generate long, detailed messages in response to user requests.",
]