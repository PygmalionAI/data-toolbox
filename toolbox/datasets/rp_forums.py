import csv
import hashlib
import logging
import os
import sys

from dataclasses import dataclass
from enum import Enum
from typing import Generator

from ..core import BaseDataset
from ..utils import enumerate_files_for

LOG = logging.getLogger(__name__)

"""
Issue:
We parse a .csv file consisting of forum posts. Each post has a thread title,
author, and message. We want to group all messages with the same thread title
together into a single thread. The issue here is that for some reason, the last
post of every thread is not added to the list of messages in the thread. That is,
if we have the following posts in the csvfile:
thread1,...,post1
thread1,...,post2
thread1,...,post3
thread2,...,post1
thread2,...,post2

Then thread1 will have posts 1 and 2 in RpThread.messages, but not 3.
thread2 will have post 1, but not 2 in RpThread.messages.
"""

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
    source_file: str


class RpForumsDataset(BaseDataset[RpThread]):
    '''Data from several different roleplay forums.'''

    def __iter__(self) -> Generator[RpThread, None, None]:
        # NOTE(11b): I had no idea this was a thing, but apparently Python's CSV
        # reader by default shits the bed if you have a field longer than 131072
        # characters. _Usually_ this means you've messed up the parsing, but in
        # our case it's actually just a massive forum post triggering this.
        # https://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072
        csv.field_size_limit(sys.maxsize)

        for path in enumerate_files_for(dataset_name="rp_forums",
                                        file_extension=".csv"):
            with open(path, "r") as file:
                reader = csv.DictReader(file, delimiter=",")
                source_file = os.path.basename(path)
                content_type = _get_rp_type_from_filename(source_file)

                # Store a history of the previous thread
                previous_thread = None
                previous_messages: list[RpMessage] = []

                for row in reader:
                    if row['thread_title'] != previous_thread:
                        # That is, if there are no previous messages
                        if previous_messages:
                            assert previous_thread is not None
                            yield RpThread(messages=previous_messages,
                                           thread_name=previous_thread,
                                           content_type=content_type,
                                           source_file=source_file)
                            previous_messages.clear()
                        previous_thread = row['thread_title']

                    # Add a message to the history of the thread
                    message = RpMessage(author=row['message_username'],
                                        message=row['message'])
                    previous_messages.append(message)

                if previous_messages:
                    # Yield the last thread in the file
                    assert previous_thread is not None
                    yield RpThread(messages=previous_messages,
                                   thread_name=previous_thread,
                                   content_type=content_type,
                                   source_file=source_file)


def _get_rp_type_from_filename(filename: str) -> RpType:
    '''
    Gets which kind of roleplaying this is based on the original file's name.
    Used to adjust the synthetic system prompt.
    '''
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
    '92dfc2e9f0fdf7efc7115e5b51ad88f01837360e9776d5e81085263b1971a9a1':
        RpType.ERP,
    'e519b14a4591a5d334d3b0e74a924296c457625cbebc3fbdc30f8810dbef3da9':
        RpType.ERP,
    '03aee36448fc81f8bae062196bad9767bfc1610c537e3a58660ba4047d49aeb5':
        RpType.ERP,
    '1bfadd54f7b41f5c2d387a4cbb9bda9342a203870e0f7be7a56a24ad3947f47a':
        RpType.ERP,
    '3d4b7c9d57643279ce091dc32e06006bc5195ab71ec3be98fef81623dcb132e7':
        RpType.ERP,
    '99131ae34901d21eca1a33ad0112fdb3f13df649c4bcf0d9e244c26273727849':
        RpType.MIXED,
    '14cc766f100cc8f1c5644d3edf822aba312d8a1c40beea7810adbd29608c9c53':
        RpType.ERP,
    'dfa38d0b1db60bf999aec14973a6919d8fbc57d217262a3877e5026f71b39d0a':
        RpType.RP,
    '795074be9881eb21bfb2ce958eda47d12e63cce1d955599d528ea257ac66f4b7':
        RpType.ERP,
    '3179b0c4ee80dc14eb3b08447d693382df2062602c40d543b1946b2ddf32daf8':
        RpType.ERP,   
}
