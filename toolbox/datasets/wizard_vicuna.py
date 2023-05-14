import json
import os
import typing as t
from dataclasses import dataclass

from toolbox.core.dataset import BaseDataset, get_path_for


@dataclass(frozen=True)
class WizardVicunaConversation:
    id: str
    human_question: str
    gpt_response: str


class WizardVicunaDataset(BaseDataset[WizardVicunaConversation]):
    '''
    Data from WizardVicuna.

    https://huggingface.co/datasets/junelee/wizard_vicuna_70k
    '''

    def __iter__(self) -> t.Generator[WizardVicunaConversation, None, None]:
        root_path = get_path_for("wizard_vicuna_70k")
        file_path = os.path.join(root_path, "wizard_vicuna_dataset.json")

        with open(file_path, "r") as file:
            data = json.load(file)
            for entry in data:
                messages = entry["conversations"]
                for idx in range(0, len(messages), 2):
                    human_message = messages[idx]
                    gpt_message = messages[idx + 1]

                    # Sanity check.
                    assert human_message["from"] == "human"
                    assert gpt_message["from"] == "gpt"

                    yield WizardVicunaConversation(
                        id=entry["id"],
                        human_question=human_message["value"],
                        gpt_response=gpt_message["value"],
                    )
