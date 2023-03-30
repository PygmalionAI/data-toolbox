import re
import typing as t

from toolbox.core.consts import PromptConstants
from toolbox.core.models import Episode, Turn
from toolbox.datasets.parlai_sdgt import ParlAiSdgtDataset
from toolbox.modules import BaseModule


class ParlAiKGM(BaseModule):
    '''Knowledge Grounding Module based on ParlAI data.'''

    def generator(self) -> t.Generator[Episode, None, None]:
        for episode in ParlAiSdgtDataset():
            prompt, _, knowledge, _, _ = re.split(r"__(end)?knowledge__",
                                                  episode.text)
            answer = episode.labels[0]

            if not knowledge.strip():
                continue

            turns: list[Turn] = []

            if episode.id in [
                    "TaskmasterSearchDialogueTeacher",
                    "GoogleSgdSearchDialogueTeacher",
            ]:
                messages = list(filter(lambda x: len(x) > 1,
                                       prompt.split("\n")))
                for idx, message in enumerate(messages):
                    # Nasty, refactor later.
                    if len(messages) % 2 == 0:
                        speaker = PromptConstants.BOT_TOKEN \
                            if idx % 2 == 0 else PromptConstants.USER_PREFIX
                    else:
                        speaker = PromptConstants.USER_PREFIX \
                            if idx % 2 == 0 else PromptConstants.BOT_TOKEN

                    turns.append(
                        Turn(
                            utterance=message,
                            speaker=speaker,
                            # All of these are marked as human so we don't use
                            # them as training labels.
                            human_speaker=True,
                        ))
            else:
                turns.append(
                    Turn(utterance=prompt,
                         speaker=PromptConstants.USER_PREFIX,
                         human_speaker=True))

            turns += [
                Turn(utterance=knowledge,
                     speaker=PromptConstants.KNOWLEDGE_PREFIX,
                     human_speaker=True),
                Turn(utterance=answer,
                     speaker=PromptConstants.BOT_TOKEN,
                     human_speaker=False),
            ]

            yield Episode(turns=turns)
