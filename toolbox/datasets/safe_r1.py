import json
import logging
import os
import typing as t
from dataclasses import dataclass

from toolbox.datasets import BaseDataset
from toolbox.utils.dataset import get_data_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InstructionExample:
    prompt: str
    response: str


class SafeR1Dataset(BaseDataset[InstructionExample]):
    '''Synthetically-generated instruction following dataset.'''

    def generator(self) -> t.Generator[InstructionExample, None, None]:
        dataset_path = get_data_path(dataset_name="safe_r1")
        for path in _enumerate_json_files(dataset_path):
            with open(path, "r", encoding="utf-8") as file:
                for line in file:
                    data = json.loads(line)
                    yield InstructionExample(
                        prompt=data["input"],
                        response=data["output"],
                    )


#
# Private helpers.
#


def _enumerate_json_files(root_path: str) -> list[str]:
    '''Returns a list of files available in the given `root_path`.'''
    items = os.listdir(root_path)

    files: list[str] = []
    for item in items:
        item_path = os.path.join(root_path, item)
        if not os.path.isfile(item_path) or not item_path.endswith(".jsonl"):
            # We only care about JSON files.
            continue

        absolute_file_path = os.path.abspath(os.path.join(root_path, item))
        files.append(absolute_file_path)

    return files
