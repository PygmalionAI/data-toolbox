import json
import typing as t
from dataclasses import dataclass

import mashumaro

from toolbox.datasets import BaseDataset
from toolbox.utils.files import enumerate_dataset_files


@dataclass(frozen=True)
class MemoryGenerationTeacherExample(mashumaro.DataClassDictMixin):
    id: str
    text: str


class ParlAiMgtDataset(BaseDataset[MemoryGenerationTeacherExample]):
    '''
    Dataset generated from ParlAI's MemoryGenerationTeacher.

    https://parl.ai/
    '''

    def generator(
            self) -> t.Generator[MemoryGenerationTeacherExample, None, None]:
        for path in enumerate_dataset_files("parlai",
                                            "MemoryGenerationTeacher",
                                            file_extension=".jsonl"):
            with open(path, "r") as file:
                for line in file:
                    data = json.loads(line)
                    yield MemoryGenerationTeacherExample.from_dict(data)
