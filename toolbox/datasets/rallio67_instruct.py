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


class Rallio67InstructDataset(BaseDataset[InstructionExample]):
    '''
    Public instruction-following dataset. Seems to be a parsed/combined version
    of other datasets.
    '''

    def generator(self) -> t.Generator[InstructionExample, None, None]:
        for path in enumerate_dataset_files("rallio67",
                                            file_extensions=[".jsonl"]):
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
                for entry in data:
                    yield InstructionExample(
                        prompt=entry[0],
                        response=entry[1],
                    )
