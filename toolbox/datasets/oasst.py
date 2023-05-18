import json
import os
import typing as t

from dataclasses import dataclass

from toolbox.core.dataset import BaseDataset, get_path_for

@dataclass(frozen=True)
class OpenAssistantDataEntry:
    conversation: list[dict]
    language: str
    tree_id: str

class OpenAssistantDataset(BaseDataset[OpenAssistantDataEntry]):
    '''
    The OpenAssistant dataset from the OpenAssistant organization - specifically, the cleaned version.
    https://huggingface.co/datasets/OpenAssistant/oasst1
    '''

    def __iter__(self) -> t.Generator[OpenAssistantDataEntry, None, None]:
        root_data_path = get_path_for("oasst")
        file_path = os.path.join(root_data_path, "2023-04-12_oasst_ready.trees.jsonl")
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                # Create a list of conversations from the different conversation paths
                for conversation in _get_list_of_dicts(entry["prompt"]):
                    yield OpenAssistantDataEntry(
                        conversation=conversation,
                        language=entry["prompt"]["lang"],
                        tree_id=entry["message_tree_id"]
                    )

# Thanks GPT-4
def _get_list_of_dicts(d):
    '''Converts nested replies into a flattened conversation list.'''
    if not d["replies"]:
        return [[{"role": d["role"], "text": d["text"]}]]

    result = []
    for reply in d["replies"]:
        for sub_reply in _get_list_of_dicts(reply):
            result.append([{"role": d["role"], "text": d["text"]}] + sub_reply)

    return result