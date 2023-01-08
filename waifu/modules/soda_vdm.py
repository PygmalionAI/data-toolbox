import typing as t

from waifu.core.consts import PromptConstants
from waifu.datasets.soda import SodaDataset
from waifu.modules import BaseModule


class SodaVDM(BaseModule):
    '''Vanilla Dialogue Module based on the SODA dataset.'''

    def generator(self) -> t.Generator[list[str], None, None]:
        for episode in SodaDataset():
            episode_messages = []
            # Grab names to be replaced by <USER> and <BOT>. This is the first and second speaker respectively
            # I'm assuming based on looking through dataset that it will always be user-bot-user-bot pattern.
            user_name = episode.speakers[0]
            bot_name = episode.speakers[1]
            
            # First, set the scenario.
            # Make sure to replace any instance of first person in conversation with user token
            replaced_narrative = episode.narrative.replace(user_name, PromptConstants.USER_TOKEN)
            scenario = f"Scenario: {replaced_narrative}"
            episode_messages.append(scenario)
            # Next, the start token
            episode_messages.append(PromptConstants.CHAT_START_TOKEN)
            
            # I am going to assume that the length of episode.speakers is the same as the length of episode.dialogue
            # Looked pretty clean to me in the data. Fuck it, TODO: account for the possibility of that happening
            for i, utterance in enumerate(episode.dialogue):
                # For now, just leave bot's name unreplaced.
                if episode.speakers[i] == user_name:
                    name = PromptConstants.USER_PREFIX
                else:
                    name = bot_name
                episode_messages.append(f"{name}: {utterance.replace(user_name, PromptConstants.USER_TOKEN)}")
            
            yield episode_messages