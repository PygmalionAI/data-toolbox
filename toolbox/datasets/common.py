'''
Common dataclasses for use in multiple datasets.
'''

from dataclasses import dataclass

@dataclass(frozen=True)
class SimpleReplyDataInstance:
    prompt: str
    generation: str

@dataclass
class MessageAndRole:
    message: str
    role: str
