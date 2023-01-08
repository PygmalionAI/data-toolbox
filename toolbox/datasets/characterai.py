import json
import logging
import os
import typing as t
from dataclasses import dataclass

from toolbox.datasets import BaseDataset
from toolbox.utils.dataset import get_data_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaiBotInfo:
    name: str
    title: str
    description: str | None
    greeting: str

    # Optional because it might be private.
    definitions: str | None

    # Useful for when several bots have the same name - we can tell them apart
    # by their external_id.
    external_id: str

    # There's also categories, but I'm ignoring them for now since I don't think
    # they'll be of much use.


@dataclass(frozen=True)
class CaiMessage:
    is_human: bool
    text: str


@dataclass(frozen=True)
class CaiChat:
    # First message is always the bot's greeting.
    messages: list[CaiMessage]
    bot: CaiBotInfo


class CharacterAiDataset(BaseDataset[CaiChat]):
    '''Dataset for CharacterAI dumps.'''

    def generator(self) -> t.Generator[CaiChat, None, None]:
        bot_id_to_info_dict = {}

        # Do a first run through all the files to load all the definitions and
        # descriptions.
        for data in _available_json_data():
            if not _is_definition_data(data):
                continue

            bot_info = _bot_info_from_dict(data["character"])
            bot_id_to_info_dict[bot_info.external_id] = bot_info

        # Now do a second pass, to actually handle chat histories/messages.
        for data in _available_json_data():
            if _is_definition_data(data):
                continue

            # Prefer grabbing bot info from a Character Editor dump, if it
            # exists. Fall back to public data otherwise.
            bot_id = data["info"]["character"]["external_id"]
            bot_info = bot_id_to_info_dict.get(
                bot_id, _bot_info_from_dict(data["info"]["character"]))

            for history_dict in data["histories"]["histories"]:
                messages = _messages_from_dict(history_dict["msgs"])
                yield CaiChat(bot=bot_info, messages=messages)


#
# Private helpers.
#


def _enumerate_json_files(root_path: str) -> list[str]:
    '''Returns a list of files available in the given `root_path`.'''
    items = os.listdir(root_path)

    files: list[str] = []
    for item in items:
        item_path = os.path.join(root_path, item)
        if not os.path.isfile(item_path) or not item_path.endswith(".json"):
            # We only care about JSON files.
            continue

        absolute_file_path = os.path.abspath(os.path.join(root_path, item))
        files.append(absolute_file_path)

    return files


def _available_json_data() -> t.Generator[dict[str, t.Any], None, None]:
    '''
    Yields all available JSON data, parsed from the files in the CharacterAI
    data folder.
    '''
    dataset_path = get_data_path(dataset_name="characterai")

    for folder in ["public", "private"]:
        folder_path = os.path.join(dataset_path, folder)
        for json_file_path in _enumerate_json_files(folder_path):
            with open(json_file_path, "r", encoding="utf-8") as json_file:
                try:
                    yield json.load(json_file)
                except json.decoder.JSONDecodeError as ex:
                    logger.error("Failed to parse %s: %s", json_file_path, ex)


def _bot_info_from_dict(info_dict: dict[str, t.Any]) -> CaiBotInfo:
    '''Builds a CaiBotInfo object from the `character` field in the JSON.'''
    return CaiBotInfo(
        name=info_dict["name"],
        title=info_dict["title"],
        # This comes in as an empty string instead of `null` in the JSON when
        # it's not defined for some reason, so we cast to None here for clarity.
        description=info_dict["description"] or None,
        greeting=info_dict["greeting"],
        definitions=info_dict.get("definition"),
        external_id=info_dict["external_id"],
    )


def _messages_from_dict(msgs_dict: list[dict[str, t.Any]]) -> list[CaiMessage]:
    '''Builds an array of messages from an entry from the `histories` JSON.'''
    messages: list[CaiMessage] = []
    for raw_message in msgs_dict:
        message = CaiMessage(
            text=raw_message["text"],
            is_human=raw_message["src"]["is_human"],
        )
        messages.append(message)
    return messages


def _is_definition_data(dict_from_json: dict[str, t.Any]) -> bool:
    '''
    Figures out whether the given dict (parsed from a JSON file) is a regular
    dump, or a dump from the Character Editor (possibly containing definitions).

    If it doesn't seem like either, raises a `ValueError` so we can discard bad
    data.
    '''
    keys = list(dict_from_json.keys())

    # Some people messed with their files so the order of the keys isn't always
    # the same, so we sort for consistency.
    keys.sort()
    if keys == ["character"]:
        return True
    elif keys == ["character", "user__username"]:
        return True
    elif keys == ["histories", "info"]:
        return False
    else:
        print(dict_from_json)
        raise ValueError(f"Unexpected keys found in CAI dump JSON file: {keys}")
