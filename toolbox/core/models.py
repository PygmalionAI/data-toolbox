from dataclasses import dataclass, field


@dataclass(frozen=True)
class Turn:
    '''Can be thought of as a message in a conversation.'''
    utterance: str
    speaker: str
    human_speaker: bool


@dataclass(frozen=True)
class Episode:
    '''Can be thought of as an entire conversation.'''
    turns: list[Turn]
    participant_personas: dict[str, str] = field(default_factory=dict)
    world_scenario: str | None = None


@dataclass(frozen=True)
class SupervisedExample:
    '''An example to be used in a supervised fine-tune.'''
    prompt: str
    response: str
