import os
from abc import ABC, abstractmethod

from datasets import Dataset, load_dataset

class TrainingData(ABC):
    def __init__(self, dataset_name: str) -> None:
        """
        The base class for a collection of data which can be used to generate training examples
        using Tasks. This class cannot and should not be instantiated directly. The role of this class
        is to take a dataset which may be present in a variety of formats (e.g. HuggingFace, CSV, JSON)
        and convert it into a HuggingFace Dataset object.
        Args:
            dataset_name (str): A shorthand name of the dataset used for generating the `identifier` field.
        """
        self.dataset_name = dataset_name
        # Set to None initially for this, to be populated by subclasses.
        # Make sure that self.dataset does not remain None after instantiation.
        self.dataset = None

    def __len__(self) -> int:
        """
        Return the number of examples in the dataset.
        """
        return len(self.dataset) if self.dataset is not None else 0

    @abstractmethod
    def to_hf_dataset(self) -> None:
        """
        Convert the data to a HuggingFace Dataset object.
        """
        pass

class ShareGptHuggingFaceData(TrainingData):
    def __init__(self, dataset_name: str, split: str = "train") -> None:
        """
        A TrainingData class which takes in a HuggingFace Dataset object and simply returns it.
        This is useful for datasets which are already in the ShareGPT format.
        Args:
            dataset_name (str): A shorthand name of the dataset used for generating the `identifier` field.
            split (str): The split of the dataset to use (e.g. "train", "test", "validation"). Default is "train".
        """
        super().__init__(self.dataset_name)
        # Dataset name is just what it'll be on HF, without the username.
        self.dataset_name = dataset_name.split("/")[-1]
        self.dataset = load_dataset(dataset_name, split=split)

        self.to_hf_dataset()

    def _add_loss_and_name(self, example: dict) -> dict:
        """
        Add the `loss` and `name` fields to the example.
        Accounts for whether they are there or not. Note that this assumes there is no name to extract from the message body
        or no special loss criteria if they are not already present. If there is, one can subclass `ShareGptHuggingFaceData`
        and override this method.
        """
        new_example = []
        for c in example['conversations']:
            if not c.get('prefix', False):
                c['name'] = ""
            if not c.get('loss', False):
                c['loss'] = c['from'] not in ['human', 'system']
        new_example.append(c)

        return {'conversations': new_example}
    
    def filter(self, fn, **kwargs) -> None:
        """
        A wrapper around `self.dataset.filter` to allow for easy filtering of the dataset.
        """
        self.dataset = self.dataset.filter(fn, **kwargs)
    
    def map(self, fn, **kwargs) -> None:
        """
        A wrapper around `self.dataset.map` to allow for easy mapping of functions to the dataset.
        """
        self.dataset = self.dataset.map(fn, **kwargs)

    def to_hf_dataset(self) -> None:
        """
        Add the `loss` and `name` fields to the HF dataset, if they are not already present.
        """
        self.dataset = self.dataset.map(
            self._add_loss_and_name,
            num_proc=os.cpu_count(),
            description=f"Converting {self.dataset_name} to internal format."
        )
