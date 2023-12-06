import logging
import os
import re

from dataclasses import dataclass
from typing import Generator

import ujson

from .common import MessageAndRole
from ..core import BaseDataset
from ..utils import get_path_for

LOG = logging.getLogger(__name__)

# I can't believe I have to do this here
HANGUL_PATTERN = re.compile(r"[\uac00-\ud7af]*")

@dataclass(frozen=True)
class TeatimeChat:
    messages: list[MessageAndRole]
    model: str
    extracted_from: str

class TeatimeDataset(BaseDataset[TeatimeChat]):
    '''
    A bunch of logs from Korean proxies. Use only the longest logs.
    https://huggingface.co/datasets/OpenLeecher/Teatime/resolve/main/all_logs_longest.json
    '''
    def __iter__(self) -> Generator[TeatimeChat, None, None]:
        root_path = get_path_for("teatime")
        file_path = os.path.join(root_path, "all_logs_longest.json")

        with open(file_path, "r", encoding="latin-1") as f:
            # Load entire file
            data = ujson.load(f)
            for entry in data:
                # Usually we don't do any processing in a dataset file, but this log
                # for some reason in particular runs into UTF-8 encoding issues
                # even though we opened it in UTF-8 format. Since TeatimeChat is
                # a frozen dataclass, we have to modify it here.
                messages = []
                for m in entry["chat"]:
                    # Check for NaN
                    if type(m["content"]) != float:
                        m["content"] = _fix_mangled_encoding(m["content"])
                    message = MessageAndRole(message=m["content"], role=m["role"])
                    messages.append(message)

                yield TeatimeChat(
                    messages=messages,
                    model=entry["model"],
                    extracted_from=entry["extracted_from"]
                )

def _fix_mangled_encoding(s: str) -> str:
    '''
    Fixes mangled encoding in a string.
    You'd think that opening a file in UTF-8 format would prevent this, but no,
    because as it turns out, some strings are encoded in Latin-1!

    '''
    # First, we find Hangul, since those blocks aren't encodable with
    # raw_unicode_escape. We have to use UTF-8 and accept that any Latin-1
    # character will become mangled.
    if HANGUL_PATTERN.search(s) is not None:
        return s.encode('utf-8').decode('utf-8')
    try:
        return s.encode('raw_unicode_escape').decode('utf-8')
    except:
        # Sometimes there are Latin-1 characters instead.
        return s.encode('raw_unicode_escape').decode('latin-1')
