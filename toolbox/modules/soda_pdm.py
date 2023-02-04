import typing as t

from toolbox.core.consts import PromptConstants
from toolbox.datasets.soda import SodaDataset
from toolbox.modules import BaseModule


class SodaPDM(BaseModule):
    '''Persona Dialogue Module based on the SODA dataset.'''

    def generator(self) -> t.Generator[list[str], None, None]:
        for episode in SodaDataset():
            episode_messages = []
            # NOTE(TG): We determine which order the speakers go on based on whether the relation is xAttr or not.
            # This is because some speakers are more abstract concepts rather than concrete names,
            # which would make them much more suitable as a bot
            if episode.relation == "xAttr":
                bot_name = episode.speakers[0]
                user_name = episode.speakers[1]
            else:
                user_name = episode.speakers[0]
                bot_name = episode.speakers[1]

            # First, we would want to set the persona.
            # However, the only acceptable description of a persona would be when episode.relation is "xAttr", since that directly describes
            # a person in the conversation.
            if episode.relation == "xAttr":
                episode_messages.append(f"{PromptConstants.pdm_prefix_for(bot_name)}: {episode.literal}")
            else:
                continue

            # Next, set the scenario.
            # Make sure to replace any instance of the person representing the user in the conversation with the user token
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
