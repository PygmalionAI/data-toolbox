import json
import logging
import os
import typing as t

from toolbox.core.dataset import BaseDataset, get_path_for
from toolbox.datasets.common import SimpleReplyDataInstance

LOG = logging.getLogger(__name__)

class ClaudeEvolInstructDataset(BaseDataset[SimpleReplyDataInstance]):
    '''
    Instructions augmented via. WizardLM's Evol-Instruct technique, answered with Claude
    https://huggingface.co/datasets/Norquinal/claude_evol_instruct_210k
    '''
    def __iter__(self) -> t.Generator[SimpleReplyDataInstance, None, None]:
        root_path = get_path_for("claude-evol")
        file_path = os.path.join(root_path, "claude_evol_instruct_210k.json")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Go through the logs and simply fetch them
            for entry in data:
                yield SimpleReplyDataInstance(
                    prompt=entry["instruction"],
                    generation=entry["output"],
                )
