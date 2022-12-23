import typing as t

from waifu.core.consts import PromptConstants
from waifu.datasets.kajiwoto import (KajiwotoDataset, generate_variants_for,
                                     replace_special_tokens_in)
from waifu.modules import BaseModule

# TODO(11b): Figure out if we can do something better instead of hardcoding a
# fake name.
BOT_PREFIX = "Person 2"


class KajiwotoVDM(BaseModule):
    '''A Vanilla Dialogue Module powered by the Kajiwoto dataset.'''

    def generator(self) -> t.Generator[str, None, None]:
        dataset = KajiwotoDataset()
        for episode in dataset:
            turns: t.List[str] = []
            for turn in episode:
                turns.append(
                    f"{PromptConstants.USER_PREFIX}: {turn.user_message}")
                turns.append(f"{BOT_PREFIX}: {turn.bot_response}")

            string = "\n".join(turns)
            processed_string = replace_special_tokens_in(string)

            for generated_string in generate_variants_for(processed_string):
                yield generated_string
