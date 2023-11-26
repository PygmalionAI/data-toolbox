from dataclasses import dataclass
from enum import Enum

class TurnKind(Enum):
    '''
    Identifies who a turn "belongs" to.
    NOTE(TG): This might be considered a 'legacy class', as it was originally
    built for only the Metharme format. Formats would have their own values for
    system, user, and model. However, we keep it for any future use that may
    require an enum (and generally just looks better in code rather than
    simple strings, IMO).
    '''
    SYSTEM = ""
    USER = ""
    MODEL = ""

@dataclass(frozen=True)
class Turn:
    '''Can be thought of as a message or interaction within a conversation.'''
    utterance: str
    kind: TurnKind
    # Provide a default name so that tasks that don't support names can work
    # with formats that require them.
    name: str = "<BOT>"
        
@dataclass(frozen=True)
class Episode:
    '''A collection of turns.'''
    turns: list[Turn]
    identifier: str
