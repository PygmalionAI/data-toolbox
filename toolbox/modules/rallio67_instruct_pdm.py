import random
import typing as t

from toolbox.core.consts import PromptConstants
from toolbox.core.models import Episode, Turn
from toolbox.datasets.rallio67_instruct import Rallio67InstructDataset
from toolbox.modules import BaseModule


class Rallio67InstructPDM(BaseModule):
    '''Persona Dialogue Module based on instruction following data.'''

    SYNTHETIC_PERSONAS = [
        "An artificial intelligence that can generate meaningful responses in plain English",
        "A helpful AI assistant", "Very intelligent",
        "A large language model (LLM) capable of answering user questions",
        "Highly knowledgeable and intelligent AI assistant",
        "Does their best to answer the user's questions",
        "Knows a lot, and always attempts to tell the truth"
    ]

    def generator(self) -> t.Generator[Episode, None, None]:
        for episode in Rallio67InstructDataset():
            turns: list[Turn] = [
                Turn(utterance=episode.prompt,
                     speaker=PromptConstants.USER_PREFIX,
                     human_speaker=True),
                Turn(utterance=episode.response,
                     speaker=PromptConstants.BOT_TOKEN,
                     human_speaker=False),
            ]

            persona = {
                PromptConstants.BOT_TOKEN:
                    self._generate_synthetic_persona_string()
            }

            yield Episode(turns=turns, participant_personas=persona)

    def _generate_synthetic_persona_string(self) -> str:
        selected_personas = random.sample(self.SYNTHETIC_PERSONAS, 5)
        random.shuffle(selected_personas)
        return ". ".join(selected_personas) + "."
