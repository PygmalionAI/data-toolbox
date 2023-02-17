import json
import os
import typing as t
from dataclasses import dataclass

import mashumaro

from toolbox.datasets import BaseDataset
from toolbox.utils.dataset import get_data_path


@dataclass(frozen=True)
class SearchDialogueGenerationTeacherExample(mashumaro.DataClassDictMixin):
    # Conversation context + injected knowledge between `__knowledge__` and
    # `__endknowledge__` tokens.
    text: str

    # The teacher is a mixture of other teachers. This field identifies which
    # specific teacher this specific example comes from.
    id: str

    # The actual response labels. Always contains a single string, it seems.
    labels: list[str]


class ParlAiSdgtDataset(BaseDataset[SearchDialogueGenerationTeacherExample]):
    '''
    Dataset generated from ParlAI's SearchDialogueGenerationTeacher.

    https://parl.ai/
    '''

    def generator(
        self
    ) -> t.Generator[SearchDialogueGenerationTeacherExample, None, None]:
        root_data_path = get_data_path("parlai")
        json_file_path = os.path.join(
            root_data_path, "SearchDialogueGenerationTeacher", "train.jsonl")

        with open(json_file_path, "r") as file:
            for line in file:
                data = json.loads(line)
                yield SearchDialogueGenerationTeacherExample.from_dict(data)
