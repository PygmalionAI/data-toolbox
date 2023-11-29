from ..core import BaseFormat, Episode, Turn, TurnKind

class MetharmeFormat(BaseFormat):
    def __init__(self) -> None:
        # Load up the lookup between TurnKind and Metharme's values for those
        self.turnkind_map = {
            TurnKind.SYSTEM: "<|system|>",
            TurnKind.USER: "<|user|>",
            TurnKind.MODEL: "<|model|>"
        }

    def apply_format(self, episode: Episode) -> Episode:
        '''Applies the format to the Episode itself.'''
        new_turns: list[Turn] = []
        for turn in episode.turns:
            new_turn = turn
            formatted = f"{self.turnkind_map[turn.kind]}{turn.utterance}"
            new_turn.utterance = formatted
            new_turns.append(new_turn)
        episode.turns = new_turns
        return episode

    def construct_dict(self, episode: Episode) -> dict:
        prompt = ""
        model_token = self.turnkind_map[TurnKind.MODEL]
        # Save the last one for the "generation" field.
        for turn in episode.turns[:-1]:
            prompt += turn.utterance
        # Then add a model token since that is not included in the 
        # generation field but rather the prompt field.
        prompt += model_token
        generation = episode.turns[-1].utterance.replace(model_token, "")
        dict_to_write = {
            "prompt": prompt,
            "generation": generation,
            "identifier": episode.identifier
        }
        return dict_to_write