import logging
import math
import os

from dataclasses import dataclass
from typing import Any, Generator, Optional

import ujson

from .common import MessageWithHumanBool
from ..core import BaseDataset
from ..utils import get_path_for

LOG = logging.getLogger(__name__)

@dataclass(frozen=True)
class PygClaudeRpConversation:
    messages: list[MessageWithHumanBool]
    user_name: str
    bot_name: str
    persona: Optional[str]

class PygClaudeRpDataset(BaseDataset[MessageWithHumanBool]):
    '''
    This dataset handles Claude-generated roleplay logs submitted to use by
    users from the Pygmalion community. Note that these are distinct from the
    (to be implemented) Claude proxy logs, which have a completely different format.
    '''

    def __iter__(self) -> Generator[MessageWithHumanBool, None, None]:
        # NOTE(TG): Maybe change the method of convo ID from number to timestamp?
        convo_num = 0
        for idx, data in enumerate(_available_json_data()):
            msg_list: list[MessageWithHumanBool] = []
            user_name = ""
            bot_name = ""

            try:
                # Check to see if the first entry is metadata: if so, we can see if a persona exists from that.
                if "chat_metadata" in data[0].keys():
                    conversation = data[1:]
                    persona = data[0]["chat_metadata"]["note_prompt"]
                else:
                    conversation = data
                    persona = ""

                for entry in conversation:
                    # Convert dictionaries to dataclasses
                    msg_list.append(
                        MessageWithHumanBool(
                            message=entry["mes"],
                            is_human=entry["is_user"]
                        )
                    )
                    if user_name == "" and entry["is_user"]:
                        user_name = entry["name"]
                    elif bot_name == "" and not entry["is_user"]:
                        bot_name = entry["name"]

                yield PygClaudeRpConversation(
                    messages=msg_list,
                    user_name=user_name,
                    bot_name=bot_name,
                    persona=persona if persona.strip() != "" else None,
                )

            except Exception as ex:
                LOG.error(f"Unable to parse data in conversation {convo_num} due to exception {ex}")

# TODO(TG): Move this to a util file?
def _available_json_data() -> Generator[list[dict[str, Any]], None, None]:
    '''
    Yields all available JSON data, parsed from the files in the Claude
    data folder.
    '''
    dataset_path = get_path_for("claude_pygsubmitted")

    for folder in ["public", "private"]:
        folder_path = os.path.join(dataset_path, folder)
        for json_file_path in _enumerate_json_files(folder_path):
            with open(json_file_path, "r", encoding="utf-8") as json_file:
                try:
                    yield [ujson.loads(line) for line in json_file]
                # TODO(TG): Fix the Unicode error more properly
                except (ujson.decoder.JSONDecodeError, UnicodeDecodeError) as ex:
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

    # Super nasty code to allow generation of Claude data with separate processes
    # so I can speed it up. Pass the "SHARD" and "TOTAL_SHARDS" environment
    # variables to operate on the different parts of the data.
    if "SHARD" not in os.environ:
        return files

    TOTAL_SHARDS = int(os.environ.get("TOTAL_SHARDS", 10))
    items_per_shard = math.floor(len(files) / TOTAL_SHARDS)

    shard = int(os.environ["SHARD"])
    file_range = (items_per_shard * shard, (items_per_shard * (shard + 1)) - 1)

    return files[file_range[0]:file_range[1]]
