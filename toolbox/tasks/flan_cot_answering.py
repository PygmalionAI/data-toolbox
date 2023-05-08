import logging
import random
import typing as t

from toolbox.core.models import Episode, Turn, TurnKind
from toolbox.core.task import BaseTask
from toolbox.datasets.flan_cot import FlanCotDataset
from toolbox.utils.prompts import generate_prompts

LOG = logging.getLogger(__name__)

class FlanCotAnsweringTask(BaseTask):
    '''
    Answers questions with chain-of-thought datasets taken from FLAN.
    Specifically, from this repo: https://github.com/google-research/FLAN/tree/main/flan/v2/cot_data
    '''
    def __iter__(self) -> t.Generator[Episode, None, None]:
        # Cache current_source for identifier string
        current_source = ""
        for entry in FlanCotDataset():
            # Identifier and counter caching
            if entry.source != current_source:
                current_source = entry.source
                counter = 1

            # Add a random system prompt and the user's question
            turns: list[Turn] = [
                Turn(
                    utterance=random.choice(SYSTEM_PROMPTS),
                    kind=TurnKind.SYSTEM
                ),
                Turn(
                    utterance=entry.prompt,
                    kind=TurnKind.USER
                )
            ]
            # Different CoT datasets have different answer formats.
            # Either they are formatted as "yes/no", a letter choice "(letter), or a proper answer.
            # Let's combine the reasoning with the answer.
            turns.append(
                Turn(
                    utterance=_construct_model_reply(entry.answer, entry.reasoning)
                )
            )

            identifier_str = f"flancot-{current_source}-{counter}"
            counter += 1

            yield Episode(turns=turns, identifier=identifier_str)

def _construct_model_reply(answer: str, reasoning: str) -> str:
    '''Combines the answer and the reasoning into a "synthetic reply".'''
    answer_mapping = {
        "yes": AFFIRMATIVES,
        "no": NEGATIVES,
        "it is impossible to tell": IMPOSSIBLE_TO_TELL
    }
    if answer in answer_mapping.keys():
        final_reply = f"{random.choice(answer_mapping[answer])}{reasoning}"
    else:
        final_reply = f"{random.choice(PREFACES)}. {reasoning}"
    return final_reply


# TODO(TG): Add more later, can't think of any prompts right now
_BASE_PROMPTS = [
    "%{Enter|Engage|Begin|Bring yourself into} question-answer mode. Given a question from the user and several options, you will answer said question correctly and give a logical explanation as to why you chose that answer. Be %{brisk|brief} in your replies.",
    "You are an %{AI|assistant} designed to logically answer %{questions|queries}. Answer %{any and all questions|every question} you get and explain how you got that answer.",
    "Your {job|role|objective} is to answer questions. Please answer any %{questions|queries} that are given to you using %{chain-of-thought|chain of thought|logical} reasoning."
]

SYSTEM_PROMPTS = generate_prompts(_BASE_PROMPTS)

AFFIRMATIVES = [
    "Yes! ",
    "Indeed. ",
    "The answer is yes. ",
    "Yes. "
]

NEGATIVES = [
    "Nope. ",
    "The answer is no. ",
    "No. ",
    "No! "
]

IMPOSSIBLE_TO_TELL = [
    "It is impossible to tell. ",
    "I cannot tell you that without further information. ",
    "From the context, it is not possible to determine the answer to that question. ",
    "Sorry, but without further context, the answer to that question is not absolute. "
]

PREFACES = [
    "",
    "The answer is ",
    "I believe the answer would be ",
    "The correct answer is "
]