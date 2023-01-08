import typing as t

from toolbox.core.consts import PromptConstants
from toolbox.datasets.kajiwoto import (KajiwotoDataset, generate_variants_for,
                                     replace_special_tokens_in)
from toolbox.modules import BaseModule


class KajiwotoVDM(BaseModule):
    '''A Vanilla Dialogue Module powered by the Kajiwoto dataset.'''

    def generator(self) -> t.Generator[list[str], None, None]:
        dataset = KajiwotoDataset()
        for episode in dataset:
            turns: t.List[str] = []
            for turn in episode:
                turns.append(
                    f"{PromptConstants.USER_PREFIX}: {turn.user_message}")
                turns.append(
                    f"{PromptConstants.BOT_TOKEN}: {turn.bot_response}")

            string = "\n".join(turns)
            processed_string = replace_special_tokens_in(string)

            for generated_string in generate_variants_for(processed_string):
                yield generated_string.split("\n")
