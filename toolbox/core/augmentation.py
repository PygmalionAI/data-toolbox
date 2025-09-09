import hashlib

from abc import ABC, abstractmethod

from datasets import Dataset

ID_HASH_LEN = 16

class Augmentation(ABC):
    def __init__(self) -> None:
        """
        The Augmentation class is designed to create (more) instruction examples from any TrainingData, without
        having to define a new Task. This can allow for creation of massive instruction datasets from a small amount
        of data. The difference between a Task and an Augmentation is that a Task allows for more data-specific
        transformations, while an Augmentation focuses on generalizing existing examples. This class cannot and should not be
        instantiated directly. Instead, subclasses should be created for each specific augmentation type.
        """
        pass

    def _generate_identifier(self, example: dict, dataset_name: str) -> str:
        """
        Generate a unique identifier for each example in the dataset.
        This is used to ensure that each example can be uniquely identified.
        """
        dataset_str = "\n".join(c['value'] for c in example['conversations'])
        # SHA-256 hash.
        example_hash = hashlib.sha256(dataset_str.encode('utf-8')).hexdigest()[:ID_HASH_LEN]
        identifier = f"{dataset_name}-augment-{example_hash}"
        # Join the identifier to the example and return it.
        return example | {'identifier': identifier}

    @abstractmethod
    def __call__(self, dataset: Dataset) -> Dataset:
        """
        Apply the augmentation to the dataset and return the augmented dataset.
        Most likely way to do this is to create new examples and concatenate them to the existing dataset.
        """
        pass
