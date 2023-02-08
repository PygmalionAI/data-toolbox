import random
import re
import typing as t

from toolbox.core.consts import PromptConstants
from toolbox.core.models import Episode, Turn
from toolbox.datasets.safe_r1 import SafeR1Dataset
from toolbox.modules import BaseModule


class SafeR1PDM(BaseModule):
    '''Persona Dialogue Module based on instruction following data.'''

    SYNTHETIC_PERSONAS = [
        "An artificial intelligence that can generate meaningful responses in plain English",
        "A helpful AI assistant", "Very intelligent", "Extremely knowledgeable",
        "A large language model (LLM) capable of answering user questions",
        "Highly knowledgeable and intelligent AI assistant",
        "Does their best to answer the user's questions",
        "Knows a lot, and always attempts to tell the truth",
        "Gives lengthy, extremely detailed responses",
        "In-depth, detailed answers"
    ]

    def generator(self) -> t.Generator[Episode, None, None]:
        for episode in SafeR1Dataset():
            messages = [
                x.strip()
                for x in episode.prompt.split("<|STK_SP|>")
                if _should_keep_message(x)
            ]
            messages.append(episode.response)
            if len(messages) < 2:
                continue

            turns: list[Turn] = []
            for message in messages:
                is_human_speaker = message.startswith("[User]")
                utterance = re.sub(r"\[(System|User|Assistant)\]", "",
                                   message).replace("<|STK_SP|>", "").strip()
                speaker = PromptConstants.USER_PREFIX \
                    if is_human_speaker else PromptConstants.BOT_TOKEN

                turns.append(
                    Turn(utterance=utterance,
                         speaker=speaker,
                         human_speaker=is_human_speaker))

            persona = {
                PromptConstants.BOT_TOKEN:
                    self._generate_synthetic_persona_string()
            }

            yield Episode(turns=turns, participant_personas=persona)

    def _generate_synthetic_persona_string(self) -> str:
        selected_personas = random.sample(self.SYNTHETIC_PERSONAS, 5)
        random.shuffle(selected_personas)
        return ". ".join(selected_personas) + "."


def _should_keep_message(message: str) -> bool:
    # Discard prompt trickery we don't want.
    if "quality: high" in message:
        return False
    if "You are a brilliant and helpful coding assistant" in message:
        return False

    # Discard empty messages.
    stripped_message = message.strip()
    if len(stripped_message) == 0:
        return False
    if stripped_message in ["[System]", "[User]", "[Assistant]"]:
        return False

    # Otherwise, looks good to keep!
    return True
