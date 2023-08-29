import json
import typing as t

from toolbox.core.dataset import BaseDataset
from toolbox.datasets.common import AlpacaLikeDataInstance
from toolbox.utils.files import enumerate_files_for

class Gpt4LlmDataset(BaseDataset[AlpacaLikeDataInstance]):
    '''
    GPT-4-LLM data.

    https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM
    '''

    def __iter__(self) -> t.Generator[AlpacaLikeDataInstance, None, None]:
        filepaths = enumerate_files_for("gpt-4-llm", file_extension="json")

        for path in filepaths:
            if "comparision_data.json" in path:
                # TODO(11b): Handle this later.
                continue

            with open(path, "r") as file:
                data = json.load(file)
                for entry in data:
                    yield AlpacaLikeDataInstance(
                        instruction=entry["instruction"],
                        input=entry["input"],
                        output=entry["output"],
                    )
