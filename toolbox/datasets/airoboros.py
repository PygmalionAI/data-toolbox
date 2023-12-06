import logging
import os

from typing import Generator

import ujson

from .common import SimpleReplyDataInstance
from ..core import BaseDataset
from ..utils import get_path_for

LOG = logging.getLogger(__name__)

class AiroborosDataset(BaseDataset[SimpleReplyDataInstance]):
    '''
    The Airoboros 1.4.1 dataset, by jondurbin.
    https://huggingface.co/datasets/jondurbin/airoboros-gpt4-1.4.1
    '''
    def __iter__(self) -> Generator[SimpleReplyDataInstance, None, None]:
        root_path = get_path_for("airoboros")
        file_path = os.path.join(root_path, "instructions.jsonl")

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                example = ujson.loads(line)
                yield SimpleReplyDataInstance(
                    prompt=example["instruction"],
                    generation=example["response"]
                )
