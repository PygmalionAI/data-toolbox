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
    # Used only for Pygmalion format
    name: str = "<BOT>"

    def as_meth_str(self) -> str:
        return f"{self.kind.value}{self.utterance}"

    def as_pyg_str(self) -> str:
        # Handle system prompt separately
        if self.kind == TurnKind.SYSTEM:
            return f"{self.name}'s Persona: {self.utterance}\n<START>\n"
        else:
            return f"\n{self.name}: {self.utterance}"

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
