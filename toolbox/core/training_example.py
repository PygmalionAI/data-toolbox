import logging
import math

from dataclasses import dataclass
from typing import Generator, Type

from .formats import BaseFormat
from .turns import Episode, Turn, TurnKind

LOG = logging.getLogger(__name__)

# TODO(TG): Allow for using a tokenizer proper rather than approximations if speed
# is not a factor.
AVG_WORD_TO_TOKEN_RATIO = 1.3

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
        formatter: Type[BaseFormat],
        target_token_count: int = 4096,
    ) -> None:
        self.episode = episode
        self.format = formatter
        self.target_token_count = target_token_count
        self.turn_order: list[TurnKind] = [
            t.kind for t in self.episode.turns
        ]

        # Run an assertion that the turn is "valid" by checking whether the
        # first term is a system prompt and that both model and user gens are
        # present.
        assert self.turn_order[0] == TurnKind.SYSTEM and set(self.turn_order) \
        == {TurnKind.SYSTEM, TurnKind.MODEL, TurnKind.USER}, f"Weird error!\n\nEpisode: {[t.utterance for t in self.episode.turns]}"

    def __iter__(self) -> Generator[TrainingExample, None, None]:
        # Modify the Episode to add in the format before calculating token counts.
        self.episode = self.format.apply_format(self.episode)
        # Calculate the token counts now starting from the system turn.
        total_tokens = 0
        if self.target_token_count != -1:
            trimmed_turns: list[Turn] = []
            for turn in self.episode.turns:
                utterance = turn.utterance
                if (tokens := (total_tokens + _token_count_for(utterance))) \
                    >= self.target_token_count:
                    # We've reached the target token count, so we stop here.
                    break
                else:
                    total_tokens += tokens
                    trimmed_turns.append(turn)

            # If the last turn is a user turn, we cut it off.
            if trimmed_turns[-1].kind == TurnKind.USER:
                trimmed_turns = trimmed_turns[:-1]

            self.episode.turns = trimmed_turns

        # Now we check if there are at least 3 turns. If not, something is
        # missing and we raise the TurnTooLargeError.
        if len(self.episode.turns) < 3:
            raise TurnTooLargeError

        # Feed the trimmed episode into the formatter again to convert to its
        # final output in the form of a dictionary.
        dict_to_write = self.format.construct_dict(self.episode)
        yield TrainingExample(
            formatted_episode=dict_to_write,
            identifier=self.episode.identifier
        )

def _token_count_for(string: str) -> int:
    '''Estimate token counts.'''
    return math.ceil(len(string.split()) * AVG_WORD_TO_TOKEN_RATIO)
