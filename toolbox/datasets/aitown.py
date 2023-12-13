import logging
import os

from typing import Generator

import ujson

from .common import MessageAndRole, MessageAndRoleConversation
from ..core import BaseDataset
from ..utils import get_path_for

LOG = logging.getLogger(__name__)

class AiTownDataset(BaseDataset[MessageAndRole]):
    '''
    The AI Town Dataset, a list of GPT-generated conversations.
    https://huggingface.co/datasets/recursal/ai-town-filtered
    We use conversations.jsonl here, not the ChatML version.
    '''
    def __iter__(self) -> Generator[MessageAndRoleConversation, None, None]:
        root_path = get_path_for("ai_town")
        file_path = os.path.join(root_path, "conversations.jsonl")

        with open(file_path, "r", encoding="utf-8") as f:
            for convo in f:
                example = ujson.loads(convo)
                conversation = [
                    MessageAndRole(
                        message=e["message"],
                        role=e["sender"]
                    ) for e in example
                ]
                yield MessageAndRoleConversation(conversation=conversation)
