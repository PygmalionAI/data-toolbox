from dataclasses import dataclass
from enum import Enum


class TurnKind(Enum):
    '''Identifies who a turn "belongs" to.'''
    SYSTEM = "<|system|>"
    USER = "<|user|>"
    MODEL = "<|model|>"


@dataclass(frozen=True)
class Turn:
    '''Can be thought of as a message or interaction within a conversation.'''
    utterance: str
    kind: TurnKind

    def as_str(self) -> str:
        return f"{self.kind.value}{self.utterance}"


@dataclass(frozen=True)
class Episode:
    '''A collection of turns.'''
    turns: list[Turn]
    identifier: str


@dataclass(frozen=True)
class TrainingExample:
    prompt: str
    generation: str
    identifier: str