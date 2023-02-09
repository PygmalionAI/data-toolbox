import typing as t

from toolbox.core.consts import PromptConstants
from toolbox.core.models import Episode, Turn
from toolbox.datasets.flan import FlanDataset
from toolbox.modules import BaseModule


class FlanVDM(BaseModule):
    '''Vanilla Dialogue Module based on the Flan dataset.'''

    def generator(self) -> t.Generator[Episode, None, None]:
        for episode in FlanDataset():
            turns: list[Turn] = [
                Turn(utterance=episode.prompt,
                     speaker=PromptConstants.USER_PREFIX,
                     human_speaker=True),
                Turn(utterance=episode.response,
                     speaker=PromptConstants.BOT_TOKEN,
                     human_speaker=False),
            ]

            yield Episode(turns=turns)
