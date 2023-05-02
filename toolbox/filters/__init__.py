import typing as t

from toolbox.filters.training_example.duplicate_filter import DuplicateFilter
from toolbox.filters.training_example.refusal_filter import RefusalFilter
from toolbox.filters.training_example_filter import TrainingExampleFilter

NAME_TO_TRAINING_EXAMPLE_FILTER_MAPPING: dict[
    str, t.Type[TrainingExampleFilter]] = {
        cls.__name__: cls for cls in [DuplicateFilter, RefusalFilter]
    }
