import typing as t

from toolbox.core.consts import PromptConstants
from toolbox.core.models import Episode, Turn
from toolbox.datasets.kajiwoto import (KajiwotoDataset, generate_variants_for,
                                       replace_special_tokens_in)
from toolbox.modules import BaseModule


class KajiwotoVDM(BaseModule):
    '''A Vanilla Dialogue Module powered by the Kajiwoto dataset.'''

    def generator(self) -> t.Generator[Episode, None, None]:
        dataset = KajiwotoDataset()
        for episode in dataset:
            turns: list[Turn] = []
            for turn in episode:
                user_turn = Turn(
                    utterance=turn.user_message,
                    speaker=PromptConstants.USER_PREFIX,
                    human_speaker=True,
                )
                bot_turn = Turn(
                    utterance=replace_special_tokens_in(turn.bot_response),
                    speaker=PromptConstants.BOT_TOKEN,
                    human_speaker=False,
                )

                turns += [user_turn, bot_turn]

            bot_message_count = int(len(turns) / 2)
            for i in range(0, bot_message_count, 2):
                idx = i + 1
                bot_turn = turns[idx]
                for variant in generate_variants_for(bot_turn.utterance):
                    augmented_turns = turns[:idx + 1].copy()
                    augmented_turns[idx] = Turn(
                        utterance=variant,
                        speaker=PromptConstants.BOT_TOKEN,
                        human_speaker=False,
                    )
                    yield Episode(turns=augmented_turns)
