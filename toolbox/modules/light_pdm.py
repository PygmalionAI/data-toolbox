import typing as t

from toolbox.core.models import Episode, Turn
from toolbox.datasets.light import LightDataset
from toolbox.modules import BaseModule
from toolbox.utils.strings import normalize_string, title_case


class LightPDM(BaseModule):
    '''Persona Dialogue Module based on the LIGHT dataset.'''

    def generator(self) -> t.Generator[Episode, None, None]:
        for episode in LightDataset():
            participant_personas = {
                title_case(a.name): a.persona for a in episode.agents
            }
            turns: t.List[Turn] = []
            turn_count = len(episode.speech)

            for idx in range(turn_count):
                character = title_case(episode.character[idx])
                speech = normalize_string(episode.speech[idx])

                # Start off with just the actual speech dialogue.
                message = speech

                # If there was an action performed in that turn, add it to the
                # string.
                #
                # NOTE(11b): Disabled for now. Adding the action like this
                # generates grammatically incorrect sentences.

                # action = episode.action[idx]
                # if action is not None:
                #     message += f" *{action}*"

                # If there was an emote in that turn, add it to the string.
                emote = episode.emote[idx]
                if emote is not None:
                    message = f"*{emote}* {message}"

                turns.append(
                    Turn(
                        utterance=message,
                        speaker=character,
                        # With LIGHT, all turns are human speakers but for the
                        # purposes of our code, only "bot" turns are used as
                        # training data so we mark everything here as a bot
                        # message so all messages are used as training
                        # examples.
                        human_speaker=False))

            yield Episode(turns=turns,
                          participant_personas=participant_personas,
                          world_scenario=episode.context[0])
