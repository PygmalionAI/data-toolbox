import json
import logging
import typing as t
from dataclasses import dataclass

import pandas as pd

from toolbox.core.dataset import BaseDataset
from toolbox.utils.files import enumerate_files_for

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class RpMessage:
    author: str
    message: str


@dataclass(frozen=True)
class RpThread:
    messages: list[RpMessage]
    thread_name: str


class RpDataset(BaseDataset[RpThread]):
    '''Loading roleplay datasets from .csv files.'''

    def __iter__(self) -> t.Generator[RpThread, None, None]:
        for path in enumerate_files_for(dataset_name="rp",
                                        file_extension=".csv"):
            df = pd.read_csv(path)
            # Store a buffer of the previous thread
            previous_thread = None
            current_thread = None
            messages = []

            for _, row in df.iterrows():
                if row['thread_title'] != previous_thread:
                    previous_thread = current_thread
                    if len(messages) != 0:
                        yield RpThread(messages=messages,
                                       thread_name=previous_thread)
                    messages = []
                current_thread = row['thread_title']

                message = RpMessage(author=row['message_username'],
                                    message=row['message'])
                messages.append(message)
                # Check for duplicate messages by same author, they indeed exist in the dataset
                if len(set(messages)) != len(messages):
                    messages = messages[:-1]
