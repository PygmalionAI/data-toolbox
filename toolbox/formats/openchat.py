from ..core import BaseFormat, Episode, TurnKind

class OpenChatFormat(BaseFormat):
    def __init__(self) -> None:
        '''
        OpenChat format.
        We don't support C-RLFT yet, but in the future that can be added.
        '''
        # Lookup between TurnKind and OpenChat's values
        self.turnkind_map = {
            TurnKind.USER: "user",
            TurnKind.MODEL: "assistant"
        }
        self.turnkind_values = {
            TurnKind.USER: 0.,
            TurnKind.MODEL: 1.,
        }

    def apply_format(self, episode: Episode) -> Episode:
        '''Applies the format to the Episode itself.'''
        # Nothing needs to be done here.
        return episode

    def construct_dict(self, episode: Episode) -> dict:
        turns = []
        # System turn goes outside the conversation.
        system_turn = episode.turns[0].utterance
        for t in episode.turns[1:]:
            turns.append({
                "role": self.turnkind_map[t.kind],
                "content": t.utterance,
                "value": self.turnkind_values[t.kind]
            })

        dict_to_write = {
            "items": turns,
            "system": system_turn,
            "identifier": episode.identifier
        }
        return dict_to_write
