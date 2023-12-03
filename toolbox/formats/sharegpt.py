from ..core import BaseFormat, Episode, Turn, TurnKind

class ShareGptFormat(BaseFormat):
    def __init__(self) -> None:
        # Lookup between TurnKind and ShareGPT's values
        self.turnkind_map = {
            TurnKind.SYSTEM: "system",
            TurnKind.USER: "human",
            TurnKind.MODEL: "gpt"
        }

    def apply_format(self, episode: Episode) -> Episode:
        '''Applies the format to the Episode itself.'''
        # Nothing needs to be done here.
        return episode

    def construct_dict(self, episode: Episode) -> dict:
        turns = []
        for t in episode.turns:
            turns.append({
                "from": self.turnkind_map[t.kind],
                "value": t.utterance
            })
        dict_to_write = {
            "conversations": turns,
            "identifier": episode.identifier
        }
        return dict_to_write
