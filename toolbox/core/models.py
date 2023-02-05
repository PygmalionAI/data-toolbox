from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    '''Can be thought of as a message in a conversation.'''
    utterance: str
    speaker: str


@dataclass(frozen=True)
class Episode:
    '''Can be thought of as an entire conversation.'''
    turns: list[Turn]
