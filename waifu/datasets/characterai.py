import json
import os
import typing as t
from dataclasses import dataclass

import mashumaro

from waifu.datasets import BaseDataset
from waifu.utils.dataset import get_data_path


@dataclass(frozen=True)
class CaiBotInfo(mashumaro.DataClassDictMixin):
    name: str
    title: str
    description: str
    greeting: str


@dataclass(frozen=True)
class CaiChat:
    # First message is the bot's greeting, the one afterwards is the user.
    messages: t.List[str]
    bot_info: CaiBotInfo


class CharacterAiDataset(BaseDataset[CaiChat]):
    '''Dataset for CharacterAI dumps.'''

    def generator(self) -> t.Generator[CaiChat, None, None]:
        for folder in _enumerate_bot_folders():
            info_path = os.path.join(folder, "info.json")
            histories_path = os.path.join(folder, "histories.json")

            with open(info_path, "r", encoding="utf-8") as info_file, \
                open(histories_path, "r", encoding="utf-8") as histories_file:
                info_json = json.load(info_file)
                histories_json = json.load(histories_file)

            bot_info = CaiBotInfo.from_dict(info_json["character"])

            for history_dict in histories_json["histories"]:
                messages = _messages_from_dict(history_dict["msgs"])
                yield CaiChat(bot_info=bot_info, messages=messages)


#
# Private helpers.
#


def _enumerate_bot_folders() -> list[str]:
    '''Returns a list of folders available in the CAI data folder.'''
    dataset_path = get_data_path(dataset_name="test_characterai_dumps")
    items = os.listdir(dataset_path)

    folders: list[str] = []
    for item in items:
        item_path = os.path.join(dataset_path, item)
        if os.path.isfile(item_path):
            # We only care about folders.
            continue

        absolute_folder_path = os.path.abspath(os.path.join(dataset_path, item))
        folders.append(absolute_folder_path)

    return folders


def _messages_from_dict(msgs_dict: list[dict[str, t.Any]]) -> list[str]:
    '''Builds an array of messages from an entry from the `histories` JSON.'''
    messages: list[str] = []
    for raw_message in msgs_dict:
        messages.append(raw_message["text"])
    return messages
