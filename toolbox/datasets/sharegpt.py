import json
import logging
import typing as t
from dataclasses import dataclass

from toolbox.datasets import BaseDataset
from toolbox.utils.files import enumerate_dataset_files

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShareGPTEpisode:
    # beautiful...
    messages: list[list[list[str]] | list[str]]


class ShareGPTDataset(BaseDataset[ShareGPTEpisode]):
    '''ChatGPT conversations shared on ShareGPT.'''

    def generator(self) -> t.Generator[ShareGPTEpisode, None, None]:
        for path in enumerate_dataset_files("share_gpt",
                                            file_extension=".json"):
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
                yield ShareGPTEpisode(messages=data)
