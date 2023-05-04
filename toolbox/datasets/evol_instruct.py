import json
import os
import typing as t

from toolbox.core.dataset import BaseDataset, get_path_for
from toolbox.datasets.gpt4llm import AlpacaLikeDataInstance


class EvolInstructDataset(BaseDataset[AlpacaLikeDataInstance]):
    '''
    WizardLM data.

    https://huggingface.co/datasets/victor123/evol_instruct_70k
    '''

    def __iter__(self) -> t.Generator[AlpacaLikeDataInstance, None, None]:
        root_path = get_path_for("evol-instruct")
        file_path = os.path.join(root_path, "alpaca_evol_instruct_70k.json")

        with open(file_path, "r") as file:
            data = json.load(file)
            for example in data:
                yield AlpacaLikeDataInstance(
                    instruction=example["instruction"],
                    input=None,
                    output=example["output"],
                )