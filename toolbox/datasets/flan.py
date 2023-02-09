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
    task: str


BAD_TASKS = [
    # Translation tasks. We don't need the model to learn to translate.
    "wmt16_translate_csen_10templates_test",
    "wmt16_translate_deen_10templates_test",
    "wmt16_translate_fien_10templates_test",
    "wmt16_translate_roen_10templates_test",
    "wmt16_translate_ruen_10templates_test",
    "wmt16_translate_tren_10templates_test",
    "wmt14_enfr_10templates_test",
]


class FlanDataset(BaseDataset[InstructionExample]):
    '''
    Processed version of the Flan v1 dataset.

    Available at HuggingFace: https://huggingface.co/datasets/Muennighoff/flan
    '''

    def generator(self) -> t.Generator[InstructionExample, None, None]:
        for path in enumerate_dataset_files("flan",
                                            subfolder="train",
                                            file_extension=".jsonl"):
            if any([task_name in path for task_name in BAD_TASKS]):
                # Skip over any tasks mentioned in `BAD_TASKS`.
                continue

            with open(path, "r", encoding="utf-8") as file:
                for line in file:
                    data = json.loads(line)
                    yield InstructionExample(
                        prompt=data["inputs"],
                        response=data["targets"],
                        task=data["task"],
                    )
