import json
import logging
import os
import typing as t

from dataclasses import dataclass
from json import JSONDecodeError

from toolbox.core.dataset import BaseDataset, get_path_for

LOG = logging.getLogger(__name__)

@dataclass(frozen=True)
class GPT4AllEntry:
    source: str
    prompt: str
    response: str

class GPT4AllDataset(BaseDataset[GPT4AllEntry]):
    '''
    The dataset used to train GPT4All, generated from answers given by GPT-3.5 to assistant prompts.
    Structured in .jsonl format.
    '''
    def __init__(self, filename: str) -> None:
        root_data_path = get_path_for("gpt4all")
        self.filepath = os.path.join(root_data_path, filename)
        super().__init__()

    def __iter__(self) -> t.Generator[GPT4AllEntry, None, None]:
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                # Unless preprocessed beforehand, the GPT4All file isn't
                # a "clean" .jsonl file. There's some binary data in there
                # that are on their own lines. If that's the case on a certain line,
                # skip over it.
                try:
                    entry = json.loads(line)
                    yield GPT4AllEntry(
                        source=entry["source"],
                        prompt=entry["prompt"],
                        response=entry["response"],
                    )
                except JSONDecodeError:
                    continue
