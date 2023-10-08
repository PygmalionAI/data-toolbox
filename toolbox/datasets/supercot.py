import json
import logging
import os
import typing as t

from toolbox.core.dataset import BaseDataset, get_path_for
from toolbox.datasets.common import AlpacaLikeDataInstance

logger = logging.getLogger(__name__)

class SuperCotDataset(BaseDataset[AlpacaLikeDataInstance]):
    '''
    The SuperCOT dataset, packed neatly into standard Alpaca format.
    https://huggingface.co/datasets/kaiokendev/SuperCOT-dataset
    '''
    def __iter__(self) -> t.Generator[AlpacaLikeDataInstance, None, None]:
        root_path = get_path_for("supercot")
        file_path = os.path.join(root_path, "filtered.json")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for entry in data:
                # "rewritten_intent" is pretty similar to just a standard input
                # and replaces the "input" field in the JSON, so just conflate
                # the two.
                try:
                    input = entry["input"]
                except KeyError:
                    input = entry["rewritten_intent"]
                yield AlpacaLikeDataInstance(
                    instruction=entry["instruction"],
                    input=input,
                    output=entry["output"]
                )
