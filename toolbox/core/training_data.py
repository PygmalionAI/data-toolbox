import os
from abc import ABC, abstractmethod

from datasets import Dataset, load_dataset

class TrainingData(ABC):
    def __init__(self) -> None:
        """
        The base class for a collection of data which can be used to generate training examples
        using Tasks. This class cannot and should not be instantiated directly. The role of this class
        is to take a dataset which may be present in a variety of formats (e.g. HuggingFace, CSV, JSON)
        and convert it into a HuggingFace Dataset object.
        Args:
            dataset_name (str): A shorthand name of the dataset used for generating the `identifier` field.
        """
        # Set to None initially for this, to be populated by subclasses.
        # Make sure that self.dataset does not remain None after instantiation.
        self.dataset_name = None
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
        super().__init__()
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
        conv = []
        for c in example['conversations']:
            name = c.get('prefix', "")
            loss = c.get('loss', "NOT FOUND") # Default to NOT FOUND so that we don't have to worry about falsy values.

            conv.append(
                {
                    'from': c['from'],
                    'value': c['value'],
                    'name': name,
                    'loss': loss if loss != "NOT FOUND" else (False if c['from'] in ['human', 'system'] else True)
                }
            )
        return {'conversations': conv}
    
    def filter(self, fn, **kwargs) -> Dataset:
        """
        A wrapper around `self.dataset.filter` to allow for easy filtering of the dataset.
        """
        return self.dataset.filter(fn, **kwargs)
    
    def map(self, fn, **kwargs) -> Dataset:
        """
        A wrapper around `self.dataset.map` to allow for easy mapping of functions to the dataset.
        """
        return self.dataset.map(fn, **kwargs)

    def to_hf_dataset(self) -> None:
        """
        Add the `loss` and `name` fields to the HF dataset, if they are not already present.
        """
        # NOTE(TG): Disabled for now, shit's broke (at least for Buzz) and I have no clue why.
        #self.dataset = self.dataset.map(
        #    self._add_loss_and_name,
        #    num_proc=os.cpu_count()
        #
        #)
        pass
