import typing as t

from waifu.datasets.characterai import CharacterAiDataset
from waifu.modules import BaseModule

USER_PREFIX = "You"


class CharacterAiPDM(BaseModule):
    '''A Persona Dialogue Module powered by CharacterAI data.'''

    def generator(self) -> t.Generator[str, None, None]:
        for chat in CharacterAiDataset():
            description_string = f"{chat.bot_info.name}'s Description: {chat.bot_info.description}"
            # Empty turn to separate description from the messages.
            turns = [description_string, ""]

            for idx, raw_message in enumerate(chat.messages):
                if idx % 2 == 0:
                    message = f"{chat.bot_info.name}: {raw_message}"
                else:
                    message = f"{USER_PREFIX}: {raw_message}"
                turns.append(message)

            yield "\n".join(turns)
