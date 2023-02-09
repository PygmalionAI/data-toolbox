import logging
import random
import typing as t

from toolbox.core.consts import PromptConstants
from toolbox.core.models import Episode, Turn
from toolbox.datasets.chain_of_thought import CoTDataset
from toolbox.modules import BaseModule

LOG = logging.getLogger(__name__)


class CoTPDM(BaseModule):
    '''
    Persona Dialogue Module based off the chain of thought datasets in FLAN.
    The original CoT datasets don't have any sort of personas in them at all, but ideally we want
    to format the data so that it fits alongside the rest of the modules.
    Therefore, we make a synthetic PDM consisting of somewhat randomly generated personas.
    '''

    SYNTHETIC_PERSONAS = [
        "Does their best to answer the user's questions",
        "Explains their answers whenever they are asked a question",
        "Logically reasons about their responses", "Intelligent", "Logical",
        "Blunt and to the point", "Short, rational answers"
    ]

    def generator(self) -> t.Generator[Episode, None, None]:
        for entry in CoTDataset():
            try:
                # Format bot's answer and persona
                bot_answer = _construct_answer(
                    answer=entry.answer,
                    chain_of_thought=entry.chain_of_thought)
                bot_persona = self._generate_synthetic_persona_string()

                # Write the human turn with the question
                human_turn = Turn(utterance=entry.question,
                                  speaker=PromptConstants.USER_TOKEN,
                                  human_speaker=True)
                # Then the bot's
                bot_turn = Turn(utterance=bot_answer,
                                speaker=PromptConstants.BOT_TOKEN,
                                human_speaker=False)
                turns: list[Turn] = [human_turn, bot_turn]
                personas = {PromptConstants.BOT_TOKEN: bot_persona}

                yield Episode(turns=turns, participant_personas=personas)
            except IndexError as ex:
                LOG.error("Error constructing episode, skipping: %s", ex)

    def _generate_synthetic_persona_string(self) -> str:
        selected_personas = random.sample(self.SYNTHETIC_PERSONAS, 5)
        random.shuffle(selected_personas)
        return ". ".join(selected_personas) + "."


# Construct many different variations of answers.
AFFIRMATIVES = [
    "Yes.", "Yep.", "Mhm.", "Oh yeah.", "Yeah.", "Indeed.", "Correct."
]
NEGATIVES = ["No.", "Nope.", "Nah."]
PUNCTUATIONS = [".", "!", "?"]


def _construct_answer(answer: str, chain_of_thought: str) -> str:
    '''Constructs a unique utterance by the bot from an answer and chain of thought'''

    # If answer is specifically "yes" or "no", add in a random stock answer from the
    # earlier defined lists
    full_answer = ""
    if answer.lower().strip() == "yes":
        full_answer = random.choice(AFFIRMATIVES)
    elif answer.lower().strip() == "no":
        full_answer = random.choice(NEGATIVES)
    else:
        # Format other answer to make it a little cleaner
        full_answer = _process_other_answer(answer)

    # Add chain of thought.
    full_answer += f" {chain_of_thought}"

    return full_answer


def _process_other_answer(original_sentence: str) -> str:
    sentence = list(original_sentence)
    sentence[0] = sentence[0].upper()
    # Add a period if answer isn't already punctuated
    if sentence[-1] not in PUNCTUATIONS:
        sentence.append(".")
    return "".join(sentence)
