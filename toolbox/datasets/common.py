"""Common data structures which can apply to multiple datasets."""
from dataclasses import dataclass

@dataclass(frozen=True)
class SimpleReplyDataInstance:
    prompt: str
    generation: str

@dataclass(frozen=True)
class AlpacaLikeDataInstance:
    instruction: str
    input: str
    output: str
