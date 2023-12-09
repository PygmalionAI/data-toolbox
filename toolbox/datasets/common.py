'''
Common dataclasses for use in multiple datasets.
'''

from dataclasses import dataclass

@dataclass
class MessageAndRole:
    message: str
    role: str

@dataclass
class MessageWithHumanBool:
    message: str
    is_human: bool

@dataclass
class SimpleReplyDataInstance:
    prompt: str
    generation: str
    