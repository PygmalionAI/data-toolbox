import csv
import hashlib
import logging
import os
import typing as t
from dataclasses import dataclass
from enum import Enum

from toolbox.core.dataset import BaseDataset
from toolbox.utils.files import enumerate_files_for

LOG = logging.getLogger(__name__)


class RpType(Enum):
    ERP = "erp"
    RP = "rp"
    MIXED = "mixed"


@dataclass(frozen=True)
class RpMessage:
    author: str
    message: str


@dataclass(frozen=True)
class RpThread:
    messages: list[RpMessage]
    thread_name: str
    content_type: RpType


class RpForumsDataset(BaseDataset[RpThread]):
    '''Data from several different roleplay forums.'''

    def __iter__(self) -> t.Generator[RpThread, None, None]:
        for path in enumerate_files_for(dataset_name="rp_forums",
                                        file_extension=".csv"):
            with open(path, "r") as file:
                reader = csv.DictReader(file, delimiter=",")
                content_type = _get_rp_type_from_path(path)

                # Store a buffer of the previous thread
                previous_thread = None
                current_thread = None
                messages: list[RpMessage] = []

                for row in reader:
                    if row['thread_title'] != previous_thread:
                        previous_thread = current_thread
                        if len(messages) != 0:
                            assert previous_thread is not None
                            yield RpThread(messages=messages,
                                           thread_name=previous_thread,
                                           content_type=content_type)

                        messages = []
                    current_thread = row['thread_title']

                    message = RpMessage(author=row['message_username'],
                                        message=row['message'])
                    messages.append(message)
                    # Check for duplicate messages by same author, they indeed exist in the dataset
                    if len(set(messages)) != len(messages):
                        messages = messages[:-1]


def _get_rp_type_from_path(path: str) -> RpType:
    '''
    Gets which kind of roleplaying this is based on the original file's name.
    Used to adjust the synthetic system prompt.
    '''
    filename = os.path.basename(path)
    sha256_digest = hashlib.sha256(filename.encode()).hexdigest()

    return SHA256_DIGEST_TO_RP_TYPE_MAP[sha256_digest]


SHA256_DIGEST_TO_RP_TYPE_MAP: dict[str, RpType] = {
    '20bc5e687f866428cc1e7ad4e500c58c0d1083f6a91e8e28950449639f7c8d21':
        RpType.MIXED,
    'c961c08eb87511193e127da59fbefb0084e325304eda86ce43ace033ad3464a3':
        RpType.ERP,
    '328f8498522ba006378a15b1bb8382278617077084afa68d865eb45edb3e2476':
        RpType.ERP,
    '5d2f252abc9008cb05e1584b77347050e309abb5cde09616d1de5645658e278a':
        RpType.ERP,
}