import hashlib

from abc import ABC, abstractmethod

from training_data import TrainingData
from datasets import Dataset

ID_HASH_LEN = 16

class Task(ABC):
    def __init__(self, dataset: TrainingData, task_type: str, **kwargs) -> None:
        """
        The Task is designed to take in a collection of data (fed in as TrainingData objects) and
        transform it into a collection of training examples. This class cannot and should not be
        instantiated directly. Instead, subclasses should be created for each specific task type.
        Args:
            dataset (TrainingData): A TrainingData object.
            task_type (str): The type of task to be generated from the data (e.g. "rp", "chat", "instruct"). This is used for the identifier.
        """
        self.dataset = dataset
        self.task_type = task_type
        # Placeholder for the shorthand alias of the task, to be set by subclasses.
        # This serves as an easy-to-remember alias for the task name.
        self.task_shorthand = ""

    def _generate_identifier(self, example: dict) -> str:
        """
        Generate a unique identifier for each example in the dataset.
        This is used to ensure that each example can be uniquely identified.
        """
        dataset_str = "\n".join(c['value'] for c in example['conversations'])
        # SHA-256 hash.
        example_hash = hashlib.sha256(dataset_str.encode('utf-8')).hexdigest()[:ID_HASH_LEN]
        identifier = f"{self.dataset.dataset_name}-{self.task_type}-{example_hash}"
        # Join the identifier to the example and return it.
        return example | {'identifier': identifier}
    
    @abstractmethod
    def _process_example(self, example: dict) -> dict:
        """
        Process a single example from the dataset and return the transformed example.
        This is where the main logic of the task will be implemented and it is called through `Dataset.map`.
        This method must be overridden by subclasses.
        Args:
            example (dict): A single example from the dataset.
        """
        pass

    @abstractmethod
    def generate_examples(self) -> Dataset:
        """
        Generate the training examples from the dataset and return them as a new Dataset object.
        This method must be overridden by subclasses.
        """
        pass
