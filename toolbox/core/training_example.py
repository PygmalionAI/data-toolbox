import logging

from dataclasses import dataclass
from typing import Generator

from turns import Episode

LOG = logging.getLogger(__name__)

class TurnTooLargeError(RuntimeError):
    pass

@dataclass
class TrainingExample:
    formatted_episode: dict
    identifier: str

class TrainingExampleGenerator:
    '''
    Converts an `Episode` into a `TrainingExample` that depends on the format.
    '''
    def __init__(
        self,
        episode: Episode,
        target_token_count: int = 4096,
        format: str = "metharme"
    ) -> None:
        self.episode = episode
        # Avoid any silly errors which can result from user not lowercasing.
        self.format = format.lower()
        # TODO(TG): The rest of this.
        pass

    def __iter__(self) -> Generator[TrainingExample, None, None]:
        raise NotImplementedError
