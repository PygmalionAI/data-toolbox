import os
import re

from datasets import load_dataset

from ...core import ShareGptHuggingFaceData

class SlimOrcaData(ShareGptHuggingFaceData):
    def __init__(self, split: str = "train") -> None:
        """
        GPT-4 generated instruct data. It is available on HuggingFace at ttps://huggingface.co/datasets/Open-Orca/SlimOrca
        Args:
            split (str): The split of the dataset to use (e.g. "train", "test", "validation"). Default is "train". This is the only split in SlimOrca.
        """
        super().__init__("Open-Orca/SlimOrca", split=split)

    def to_hf_dataset(self) -> None:
        """
        Add the `loss` and `name` fields to the HF dataset, if they are not already present.
        Also removes the 'weight' field from each conversation turn if it exists.
        """
        # self._add_loss_and_name actually does work for SlimOrca. Need to test for other datasets.
        self.dataset = self.dataset.map(
            self._add_loss_and_name,
            num_proc=os.cpu_count()
        )
        # Remove 'weight' field if it exists in any conversation turn.
        def remove_weight(example: dict) -> dict:
            for c in example['conversations']:
                if 'weight' in c:
                    del c['weight']
            return example

        self.dataset = self.dataset.map(
            remove_weight,
            num_proc=os.cpu_count()
        )