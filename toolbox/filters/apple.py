import logging
import os

from datasets import Dataset

from ..core import Filter

LOG = logging.getLogger("AppleFilter")

class AppleFilter(Filter):
    FILTER_SHORTHAND = "apple_filter"
    def __init__(self) -> None:
        """
        A test Filter which removes any examples which contain the word "apple" in any of the messages.
        """
        super().__init__()

    def __call__(self, dataset: Dataset) -> Dataset:
        """
        Apply the test filter to the dataset.
        """
        orig_dataset_len = len(dataset)
        dataset = dataset.filter(
            lambda x: all('apple' not in c['value'].lower() for c in x['conversations']),
            num_proc=os.cpu_count(),
        )
        LOG.info(f"Removed {orig_dataset_len - len(dataset)} examples from dataset.")

        return dataset
