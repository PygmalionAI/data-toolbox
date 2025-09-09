import hashlib

from abc import ABC, abstractmethod

from datasets import Dataset

ID_HASH_LEN = 16

class Task(ABC):
    def __init__(self, dataset: Dataset, task_type: str) -> None:
        """
        The Task is designed to take in a collection of data (fed in as TrainingData objects) and
        transform it into a collection of training examples. This class cannot and should not be
        instantiated directly. Instead, subclasses should be created for each specific task type.
        """
        self.dataset = dataset
        self.task_type = task_type

    def _generate_identifier(self, example: dict) -> str:
        """
        Generate a unique identifier for each example in the dataset.
        This is used to ensure that each example can be uniquely identified.
        """
        dataset_str = "\n".join(c['value'] for c in example['conversations'])
        # SHA-256 hash.
        example_hash = hashlib.sha256(dataset_str.encode('utf-8')).hexdigest()[:ID_HASH_LEN]
        identifier = f"{self.dataset.dataset_name}-augment-{example_hash}"
        # Join the identifier to the example and return it.
        return example | {'identifier': identifier}

    @abstractmethod
    def generate_examples(self) -> Dataset:
        """
        Generate the training examples from the dataset and return them as a new Dataset object.
        """
        pass
