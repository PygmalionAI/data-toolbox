import json
import logging
import os
import typing as t

from dataclasses import dataclass

from toolbox.core.dataset import BaseDataset, get_path_for

LOG = logging.getLogger(__name__)

@dataclass(frozen=True)
class Airoboros2DataInstance:
    instruction: str
    response: str
    system_prompt: str
    category: str

class Airoboros2Dataset(BaseDataset[Airoboros2DataInstance]):
    '''
    Instructions from Airoboros 2.2.1
    https://huggingface.co/datasets/jondurbin/airoboros-2.2.1/
    '''
    def __iter__(self) -> t.Generator[Airoboros2DataInstance, None, None]:
        root_path = get_path_for("airoboros2")
        file_path = os.path.join(root_path, "instructions.jsonl")

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                yield Airoboros2DataInstance(
                    instruction=entry["instruction"],
                    response=entry["response"],
                    system_prompt=entry["system"],
                    category=entry["category"],
                )
