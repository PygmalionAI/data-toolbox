import typing as t

from toolbox.core.consts import PromptConstants
from toolbox.datasets.kajiwoto import (KajiwotoDataset, generate_variants_for,
                                       replace_special_tokens_in)
from toolbox.modules import BaseModule
from toolbox.utils.strings import uppercase


class KajiwotoPDM(BaseModule):
    '''A Persona Dialogue Module powered by the Kajiwoto dataset.'''

    def generator(self) -> t.Generator[str, None, None]:
        dataset = KajiwotoDataset()
        for episode in dataset:
            turns: list[str] = []

            metadata = dataset.get_metadata_for_bot(episode[0].bot_id)

            # `metadata.personalities` is in a format like: `[["friendly", "20.32"]]`
            # but we want that "phrased" closer to natural language, so we build
            # `persona_string` to take care of that.
            personality_descriptors = [x[0] for x in metadata.personalities]
            persona_string = ". ".join(
                [uppercase(x) for x in personality_descriptors]) + "."

            description_string = metadata.description.replace("\n",
                                                              " ").replace(
                                                                  "  ", " ")
            turns.append(
                f"{PromptConstants.pdm_prefix_for(PromptConstants.BOT_TOKEN)}: {description_string}\n{persona_string}"
            )

            # Empty turn to have a line break separating description/persona
            # and the actual messages.
            turns.append("")

            for turn in episode:
                turns.append(
                    f"{PromptConstants.USER_PREFIX}: {turn.user_message}")
                turns.append(
                    f"{PromptConstants.BOT_TOKEN}: {turn.bot_response}")

            string = "\n".join(turns)
            processed_string = replace_special_tokens_in(string)

            for generated_string in generate_variants_for(processed_string):
                yield generated_string
