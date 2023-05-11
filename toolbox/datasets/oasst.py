import json
import os
import typing as t

from dataclasses import dataclass

from toolbox.core.dataset import BaseDataset, get_path_for

# Thanks GPT-4
def get_list_of_dicts(d):
    '''Converts nested replies into a flattened conversation list.'''
    if not d["replies"]:
        return [[{"role": d["role"], "text": d["text"]}]]

    result = []
    for reply in d["replies"]:
        for sub_reply in get_list_of_dicts(reply):
            result.append([{"role": d["role"], "text": d["text"]}] + sub_reply)

    return result

@dataclass(frozen=True)
class OpenAssistantDataEntry:
    conversation: list[dict]
    language: str
    tree_id: str

class OpenAssistantDataset(BaseDataset[OpenAssistantDataEntry]):
    '''
    The OpenAssistant dataset. Add more here.

    Params:
    kept_languages: The list of languages in OpenAssist to keep in the final dataset. Set to None to keep all languages.
    '''

    def __iter__(self) -> t.Generator[OpenAssistantDataEntry, None, None]:
        root_data_path = get_path_for("oasst")
        file_path = os.path.join(root_data_path, "2023-04-12_oasst_ready.trees.jsonl")
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                # Create a list of conversations from the different conversation paths
                for conversation in get_list_of_dicts(entry["prompt"]):
                    yield OpenAssistantDataEntry(
                        conversation=conversation,
                        language=conversation[0]["lang"],
                        tree_id=entry["message_tree_id"]
                    )
