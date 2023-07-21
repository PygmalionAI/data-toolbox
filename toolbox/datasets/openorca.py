import logging
import os
import typing as t
from dataclasses import dataclass

import pandas as pd

from toolbox.core.dataset import BaseDataset
from toolbox.utils.files import enumerate_files_for

LOG = logging.getLogger(__name__)

@dataclass(frozen=True)
class OpenOrcaEntry:
    id: str
    system_prompt: str
    question: str
    response: str

class OpenOrcaDataset(BaseDataset[OpenOrcaEntry]):
    '''The OpenOrca dataset.'''
    def __iter__(self) -> t.Generator[OpenOrcaEntry, None, None]:
        # We have this so that one can use GPT-4 OpenOrca, 3.5 OpenOrca, or both
        for path in enumerate_files_for(dataset_name="openorca", file_extension=".parquet"):
            df = pd.read_parquet(path)
            for idx in df.index:
                yield OpenOrcaEntry(
                    id=df["id"][idx],
                    system_prompt=df["system_prompt"][idx],
                    question=df["question"][idx],
                    response=df["response"][idx]
                )
