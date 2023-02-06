import typing as t

from toolbox.core.consts import PromptConstants
from toolbox.core.models import Episode, Turn
from toolbox.datasets.soda import SodaDataset
from toolbox.modules import BaseModule


class SodaPDM(BaseModule):
    '''Persona Dialogue Module based on the SODA dataset.'''

    def generator(self) -> t.Generator[Episode, None, None]:
        for episode in SodaDataset():
            if episode.relation != "xAttr":
                # Only "xAttr" relations are usable as persona data, so we skip
                # over any other kinds.
                continue

            bot_name = episode.speakers[0]
            user_name = episode.speakers[1]
            turns: list[Turn] = []
            personas = {bot_name: episode.literal}

            # Make sure to replace any instance of the person representing the
            # user in the conversation with the user token.
            replaced_narrative = episode.narrative.replace(
                user_name, PromptConstants.USER_TOKEN)
            world_scenario = replaced_narrative

            for idx, raw_message in enumerate(episode.dialogue):
                speaker = PromptConstants.USER_PREFIX \
                    if episode.speakers[idx] == user_name else bot_name
                human_speaker = speaker == PromptConstants.USER_PREFIX
                utterance = raw_message.replace(user_name,
                                                PromptConstants.USER_TOKEN)
                turns.append(
                    Turn(utterance=utterance,
                         speaker=speaker,
                         human_speaker=human_speaker))

            yield Episode(turns=turns,
                          participant_personas=personas,
                          world_scenario=world_scenario)
