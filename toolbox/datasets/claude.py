import json
import logging
import math
import os
import typing as t

from dataclasses import dataclass

from toolbox.core.dataset import BaseDataset, get_path_for

LOG = logging.getLogger(__name__)

@dataclass(frozen=True)
class ClaudeMessage:
    message: str
    is_user: bool

@dataclass(frozen=True)
class ClaudeConversation:
    messages: list[ClaudeMessage]
    user_name: str
    bot_name: str
    convo_id: int

class ClaudeDataset(BaseDataset[ClaudeMessage]):
    '''Dataset for user-submitted Claude logs'''

    def __iter__(self) -> t.Generator[ClaudeMessage, None, None]:
        # NOTE(TG): Maybe change the method of convo ID from number to timestamp?
        convo_num = 0
        for data in _available_json_data():
            msg_list: list[ClaudeMessage] = []
            user_name = ""
            bot_name = ""

            for entry in data:
                # Convert dictionaries to dataclasses
                msg_list.append(
                    ClaudeMessage(
                        message=entry["mes"],
                        is_user=entry["is_user"]
                    )
                )
                if user_name == "" and entry["is_user"]:
                    user_name = entry["name"]
                elif bot_name == "" and not entry["is_user"]:
                    bot_name = entry["name"]

            yield ClaudeConversation(
                messages=msg_list,
                user_name=user_name,
                bot_name=bot_name,
                convo_id=convo_num,
            )

            convo_num += 1

def _available_json_data() -> t.Generator[list[dict[str, t.Any]], None, None]:
    '''
    Yields all available JSON data, parsed from the files in the Claude
    data folder.
    '''
    dataset_path = get_path_for("claude")

    for folder in ["public", "private"]:
        folder_path = os.path.join(dataset_path, folder)
        for json_file_path in _enumerate_json_files(folder_path):
            with open(json_file_path, "r", encoding="utf-8") as json_file:
                try:
                    yield [json.loads(line) for line in json_file]
                # TODO(TG): Fix the Unicode error more properly
                except (json.decoder.JSONDecodeError, UnicodeDecodeError) as ex:
                    LOG.error("Failed to parse %s: %s", json_file_path, ex)

def _enumerate_json_files(root_path: str) -> list[str]:
    '''Returns a list of files available in the given `root_path`.'''
    # TODO(11b): Implement the sharding logic out in the util, and get rid of
    # this function.

    items = os.listdir(root_path)

    files: list[str] = []
    for item in items:
        item_path = os.path.join(root_path, item)
        if not os.path.isfile(item_path) or not item_path.endswith(".jsonl"):
            # We only care about JSON files.
            continue

        absolute_file_path = os.path.abspath(os.path.join(root_path, item))
        files.append(absolute_file_path)

    # Super nasty code to allow generation of CAI data with separate processes
    # so I can speed it up. Pass the "SHARD" and "TOTAL_SHARDS" environment
    # variables to operate on the different parts of the data.
    if "SHARD" not in os.environ:
        return files

    TOTAL_SHARDS = int(os.environ.get("TOTAL_SHARDS", 10))
    items_per_shard = math.floor(len(files) / TOTAL_SHARDS)

    shard = int(os.environ["SHARD"])
    file_range = (items_per_shard * shard, (items_per_shard * (shard + 1)) - 1)

    return files[file_range[0]:file_range[1]]
