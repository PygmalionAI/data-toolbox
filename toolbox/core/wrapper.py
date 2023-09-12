from abc import ABC, abstractmethod
from toolbox.core.models import Turn, TurnKind

class TurnWrapper(ABC):
    def __init__(self, turn: Turn) -> None:
        '''Abstract wrapper for the purpose of easily constructing examples.'''
        self.turn = turn
        # Make accessing the values of Turn easier
        self.utterance = turn.utterance
        self.kind = turn.kind
        self.name = turn.name

    @abstractmethod
    def as_str(self) -> str:
        '''Convert a turn into a training example'''
        raise NotImplementedError
    
    @abstractmethod
    def get_model_turn(self) -> str:
        '''Get the model turn portion of the turn'''
        raise NotImplementedError
    
class MetharmeWrapper(TurnWrapper):
    def __init__(self, turn: Turn) -> None:
        super().__init__(turn)

    def as_str(self) -> str:
        return f"{self.kind.value}{self.utterance}"
    
    def get_model_turn(self) -> str:
        return TurnKind.MODEL.value
    
class PygmalionWrapper(TurnWrapper):
    def __init__(self, turn: Turn) -> None:
        super().__init__(turn)

    def as_str(self) -> str:
        if self.kind == TurnKind.SYSTEM:
            return f"{self.name}'s Persona: {self.utterance}\n<START>"
        else:
            return f"{self.name}: {self.utterance}"
    
    def get_model_turn(self) -> str:
        return f"\n{self.name}: "
    
class AlpacaWrapper(TurnWrapper):
    def __init__(self, turn: Turn) -> None:
        super().__init__(turn)
        self.kind_map: [TurnKind, str] = {
            TurnKind.SYSTEM: "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:",
            TurnKind.USER: "### Input:",
            TurnKind.MODEL: "### Response:"
        }

    def as_str(self) -> str:
        return f"{self.kind_map[self.kind]}\n{self.utterance}\n\n"
    
    def get_model_turn(self) -> str:
        return f"{self.kind_map[TurnKind.MODEL]}\n"
    
class MinimalAlpacaWrapper(TurnWrapper):
    def __init__(self, turn: Turn) -> None:
        super().__init__(turn)
    
    def as_str(self) -> str:
        # System prompt and user are under the same block
        if self.kind != TurnKind.MODEL:
            return f"### Instruction:\n{self.utterance}\n"
        else:
            return f"### Response:\n{self.utterance}\n"
        
    def get_model_turn(self) -> str:
        return f"### Response:\n"
    
class HenkpacaWrapper(TurnWrapper):
    def __init__(self, turn: Turn) -> None:
        super().__init__(turn)

    def as_str(self) -> str:
        if self.kind == TurnKind.SYSTEM:
            return f"### Instruction:\n{self.utterance}\n### Response:\n"
        else:
            return f"{self.name}: {self.utterance}\n"
        
    def get_model_turn(self) -> str:
        return f"{self.name}: "

WRAPPER_MAP: dict[str, TurnWrapper] = {
    "metharme": MetharmeWrapper,
    "pygmalion": PygmalionWrapper,
    "alpaca": AlpacaWrapper,
    "minimal_alpaca": MinimalAlpacaWrapper,
    "henkpaca": HenkpacaWrapper,
}
