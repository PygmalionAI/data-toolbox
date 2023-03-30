import json
import logging
import os
import typing as t
from dataclasses import dataclass

from toolbox.core.dataset import BaseDataset
from toolbox.utils.files import enumerate_files_for

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShareGptEpisode:
    # beautiful...
    messages: list[list[list[str]] | list[str]]
    source_file: str


class ShareGptDataset(BaseDataset[ShareGptEpisode]):
    '''ChatGPT conversations shared on ShareGPT.'''

    def __iter__(self) -> t.Generator[ShareGptEpisode, None, None]:
        for path in enumerate_files_for(dataset_name="sharegpt",
                                        file_extension=".json"):
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
                source_file = os.path.basename(path).replace(".json", "")
                yield ShareGptEpisode(messages=data, source_file=source_file)
