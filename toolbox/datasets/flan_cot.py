import csv
import logging
import os
import typing as t

from dataclasses import dataclass

from toolbox.core.dataset import BaseDataset
from toolbox.utils.files import enumerate_files_for

LOG = logging.getLogger(__name__)

@dataclass(frozen=True)
class FlanCotEntry:
    prompt: str
    answer: str
    reasoning: str
    source: str

class FlanCotDataset([BaseDataset[FlanCotEntry]]):
    '''A collection of chain-of-thought datasets used in the FLAN dataset.'''
    def __iter__(self) -> t.Generator[FlanCotEntry, None, None]:
        for path in enumerate_files_for(dataset_name="flan_cot", file_extension=".tsv"):
            with open(path, "r", encoding="utf-8") as file:
                reader = csv.reader(file, delimiter="\t")
                for row in reader:
                    entry = row.split("\t")
                    return FlanCotEntry(
                        prompt=entry[0],
                        answer=entry[1],
                        reasoning=entry[2],
                        source=os.path.basename(path).split(".")[0]
                    )
