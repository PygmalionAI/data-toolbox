import logging
import random
import typing as t

from toolbox.core.models import Episode, Turn, TurnKind
from toolbox.core.task import BaseTask
from toolbox.datasets.dolly import DollyDataset
from toolbox.utils.prompts import generate_prompts

LOG = logging.getLogger(__name__)

class DollyGuessTheInstructionTask(BaseTask):
    '''
    Given an answer and possibly context, task the AI to generate a proper instruction or question for it.
    Heavily inspired by "Guess the Instruction! Flipped Learning Makes Language Models Stronger Zero-Shot Learners"
    Paper: https://arxiv.org/abs/2210.02969 | Github: https://github.com/seonghyeonye/Flipped-Learning/tree/master
    '''
    def __iter__(self) -> t.Generator[Episode, None, None]:
        for i, entry in enumerate(DollyDataset()):
            turns: list[Turn] = [
                Turn(
                    utterance=random.choice(SYSTEM_PROMPTS),
                    kind=TurnKind.SYSTEM
                )
            ]
            # Construct user prompt
            user_prompt = random.choice(USER_PROMPTS)
            user_prompt = user_prompt.replace("<INFO>", entry.output)
            if entry.input != "":
                context = random.choice(CONTEXT_PREFIXES) + entry.input
                user_prompt = user_prompt.replace("<CONTEXT>", context)
            else:
                user_prompt = user_prompt.replace("<CONTEXT>", "")

            turns.append(Turn(utterance=user_prompt, kind=TurnKind.USER))
            turns.append(Turn(utterance=entry.instruction, kind=TurnKind.MODEL))
            yield Episode(turns, identifier=f"dolly-{i}")

SYSTEM_PROMPTS = [
    "Consider Assistant, a large language model (LLM). It responds to user requests as truthfully as it can.",
    "You are a large language model trained to act as an assistant. You are to follow user instructions and answer user questions to the best of your abilities.",
    "Enter assistant mode. In this mode, you will follow instructions and respond with helpful responses.",
    "You are now in assistant mode. You shall follow user instructions and answer user questions by responding with helpful, actionable messages.",
    "Assistant, engage instruction following and question answering mode. You are bound to generating text, and cannot perform any other actions.",
    "Consider Assistant, a LLM trained to follow user instructions and answer questions.",
    "Enter 'guess the instruction' mode. Given a response and possibly context, you are tasked with generating the instruction/question that could be applicable to be answered by the response.",
    "You're an LLM. Given pieces of information, your job is to come up with an instruction that fits with the information. Be brisk in your replies."
]

_BASE_USER_PROMPTS = [
    """%{Question:|Here's a question for you:|I'm gonna ask you this.|Here's a question.} <INFO> <CONTEXT>%{
        |
        
        }What is %{an|the} instruction that goes with that piece of info?""",
    """Guess the %{question|instruction} given this answer: <INFO> <CONTEXT>""",
    """Here is %{some information|a piece of text} that corresponds to what an AI would generate in response to being given an instruction.
    \"<INFO\"> <CONTEXT>
    What would have been the %{question|instruction} for %{this|that}?""",
    """ok here: <INFO>
    <CONTEXT>
    come up with %{the question|the thing i would've asked you} please"""
]

USER_PROMPTS = generate_prompts(_BASE_USER_PROMPTS)

CONTEXT_PREFIXES = ["Context: ", "You might want to know this: ", "\nHere's some further information:\n", "", "\n"]
