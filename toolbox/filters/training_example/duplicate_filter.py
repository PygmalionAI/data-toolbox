import hashlib

from toolbox.core.training_example import TrainingExample
from toolbox.filters.training_example_filter import TrainingExampleFilter


class DuplicateFilter(TrainingExampleFilter):
    '''Filters out training examples which are exact duplicates.'''

    def __init__(self) -> None:
        super().__init__()

        self.seen_hashes: set[str] = set()

    def should_keep(self, example: TrainingExample) -> bool:
        serialized_example = example.prompt + example.generation
        example_hash = _calculate_hash_for(serialized_example)
        if example_hash in self.seen_hashes:
            return False

        self.seen_hashes.add(example_hash)
        return True


def _calculate_hash_for(text: str) -> str:
    return hashlib.sha512(text.encode("utf-8")).hexdigest()
