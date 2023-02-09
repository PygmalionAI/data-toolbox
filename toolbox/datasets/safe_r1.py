import json
import logging
import typing as t
from dataclasses import dataclass

from toolbox.datasets import BaseDataset
from toolbox.utils.files import enumerate_dataset_files

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InstructionExample:
    prompt: str
    response: str


class SafeR1Dataset(BaseDataset[InstructionExample]):
    '''Synthetically-generated instruction following dataset.'''

    def generator(self) -> t.Generator[InstructionExample, None, None]:
        for path in enumerate_dataset_files("safe_r1", file_extension=".jsonl"):
            with open(path, "r", encoding="utf-8") as file:
                for line in file:
                    data = json.loads(line)
                    yield InstructionExample(
                        prompt=data["input"],
                        response=data["output"],
                    )
