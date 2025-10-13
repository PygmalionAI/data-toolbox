from abc import ABC, abstractmethod

from datasets import Dataset

class Filter(ABC):
    FILTER_SHORTHAND = "UNSET_SHORTHAND" # Subclasses should override this with a shorthand name.
    
    def __init__(self) -> None:
        """
        The Filter class is designed to filter out low-quality or unwanted examples from a Dataset.
        This class cannot and should not be instantiated directly. Instead, subclasses should be created
        for each specific filter type.
        """
        pass

    @abstractmethod
    def __call__(self, dataset: Dataset) -> Dataset:
        """
        Apply the filter to the dataset and return the filtered dataset.
        This should likely use the `Dataset.filter` method internally.
        """
        raise NotImplementedError("Subclasses are required to implement the __call__ method.")
