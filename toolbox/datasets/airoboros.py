import json
import logging
import os
import typing as t

from toolbox.core.dataset import BaseDataset, get_path_for
from toolbox.datasets.common import SimpleReplyDataInstance

logger = logging.getLogger(__name__)

class AiroborosDataset(BaseDataset[SimpleReplyDataInstance]):
    def __iter__(self) -> t.Generator[SimpleReplyDataInstance, None, None]:
        root_path = get_path_for("airoboros")
        file_path = os.path.join(root_path, "instructions.jsonl")

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line_entry = json.loads(line)
                yield SimpleReplyDataInstance(
                    prompt=line_entry["instruction"],
                    generation=line_entry["response"]
                )
