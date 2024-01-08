import logging
import os

from dataclasses import dataclass
from typing import Generator

import pandas as pd

from .common import MessageAndRole
from ..core import BaseDataset
from ..utils import get_path_for

LOG = logging.getLogger(__name__)

@dataclass(frozen=True)
class NoRobotsConversation:
    conversation: list[MessageAndRole]
    prompt_id: str
    category: str

class NoRobotsDataset(BaseDataset[NoRobotsConversation]):
    '''
    No Robots, a human-created instruction dataset.
    We use only the train split here.
    NOTE(TG): We have to switch over to pandas for everything in general, for
    much faster performance.
    https://huggingface.co/datasets/HuggingFaceH4/no_robots
    '''
    def __iter__(self) -> Generator[NoRobotsConversation, None, None]:
        root_path = get_path_for("no_robots")
        file_path = os.path.join(root_path, "train_sft-00000-of-00001-8aba5401a3b757f5.parquet")

        df = pd.read_parquet(file_path)
        for _, row in df.iterrows():
            messages = row["messages"]
            conversation = [
                MessageAndRole(
                    message=m["content"],
                    role=m["role"]
                ) for m in messages
            ]

            yield NoRobotsConversation(
                conversation=conversation,
                prompt_id=row["prompt_id"],
                category=row["category"]
            )
