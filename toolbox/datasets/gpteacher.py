import json
import os
import typing as t

from toolbox.core.dataset import BaseDataset, get_path_for
from toolbox.datasets.gpt4llm import \
    AlpacaLikeDataInstance  # not the greatest import...


class GpTeacherDataset(BaseDataset[AlpacaLikeDataInstance]):
    '''
    GPTeacher data.

    https://github.com/teknium1/GPTeacher
    '''

    def __iter__(self) -> t.Generator[AlpacaLikeDataInstance, None, None]:
        path_to_root_folder = get_path_for("gpteacher")
        for desired_filename in DESIRED_FILES:
            path = os.path.join(path_to_root_folder, desired_filename)
            with open(path, "r") as file:
                data = json.load(file)
                for entry in data:
                    yield AlpacaLikeDataInstance(
                        instruction=entry["instruction"],
                        input=entry["input"],
                        output=entry["response"],
                    )


DESIRED_FILES = [
    "Instruct/gpt4-instruct-similarity-0.9-dataset.json",
    "Roleplay/roleplay-similarity_0.9-instruct-dataset.json",
    "Toolformer/toolformer-similarity-0.9-dataset.json",
]