import logging
import os

from dataclasses import dataclass
from typing import Generator

import ujson

from ..core import BaseDataset
from ..utils import get_path_for

LOG = logging.getLogger(__name__)

@dataclass(frozen=True)
class OpenCaiCharacter:
    name: str
    description: str

@dataclass(frozen=True)
class OpenCaiMessage:
    author: str
    message: str
    is_bot: bool

@dataclass(frozen=True)
class OpenCaiConversation:
    characters: list[OpenCaiCharacter]
    conversation: list[OpenCaiMessage]
    scene: str
    summary: str
    tags: str

class OpenCaiDataset(BaseDataset[OpenCaiMessage]):
    '''
    A collection of Discord roleplay logs scraped from multiple servers.
    https://huggingface.co/datasets/Norquinal/OpenCAI
    '''
    def __iter__(self) -> Generator[OpenCaiConversation, None, None]:
        root_path = get_path_for("opencai")
        file_path = os.path.join(root_path, "opencai_rp.json")

        with open(file_path, "r", encoding="utf-8") as f:
            # Load the file.
            data = ujson.load(f)
            # Iterate through the data.
            for conv in data:
                characters = [
                    OpenCaiCharacter(
                        name=e["bot_name"],
                        description=e["bot_description"]
                    ) for e in conv["characters"]
                ]
                conversation = [
                    OpenCaiMessage(
                        author=e["author"],
                        message=e["message"],
                        is_bot=e["is_bot"]
                    ) for e in conv["conversations"]
                ]
                yield OpenCaiConversation(
                    characters=characters,
                    conversation=conversation,
                    scene=conv["scene"],
                    summary=conv["summary"],
                    tags=conv["chatTags"]
                )
                