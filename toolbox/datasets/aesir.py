import logging
import os

from typing import Generator

import ujson

from .common import MessageAndRole, MessageAndRoleConversation
from ..core import BaseDataset
from ..utils import get_path_for

LOG = logging.getLogger(__name__)

class AesirDataset(BaseDataset[MessageAndRole]):
    '''
    A roleplay dataset.
    '''
    def __iter__(self) -> Generator[MessageAndRoleConversation, None, None]:
        root_path = get_path_for("aesir")
        file_path = os.path.join(root_path, "aesir-rpg-charcards.json")

        with open(file_path, "r", encoding="utf-8") as f:
            # Load the file.
            data = ujson.load(f)
            # Iterate through the data.
            for conv in data:
                convo = [
                    MessageAndRole(
                        message=e["value"],
                        role=e["from"]
                    ) for e in conv["conversations"]
                ]
                yield MessageAndRoleConversation(conversation=convo)
