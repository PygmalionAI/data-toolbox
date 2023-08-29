import json
import logging
import os
import typing as t

from dataclasses import dataclass

from toolbox.core.dataset import BaseDataset, get_path_for

@dataclass(frozen=True)
class ClaudeMultiround:
    conversation: list[dict[str, str]]
    id: str

class ClaudeInstructDataset(BaseDataset[ClaudeMultiround]):
    '''
    Logs taken from synthetically-generated instruction chats with Claude.
    https://huggingface.co/datasets/Norquinal/claude_multiround_chat_30k
    '''
    def __iter__(self) -> t.Generator[ClaudeMultiround, None, None]:
        root_path = get_path_for("claude-multiround")
        file_path = os.path.join(root_path, "claude_multiround_chat_30k.json")

        with open(file_path, "r", encoding="utf-8") as f:
            logs = json.load(f)
            # Go through the logs and simply fetch them
            for round in logs:
                yield ClaudeMultiround(
                    conversation=round["conversations"],
                    id=round["id"],
                )
