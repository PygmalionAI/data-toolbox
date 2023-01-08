import typing as t

from toolbox.core.consts import PromptConstants
from toolbox.datasets.light_dialogue import LightDialogueDataset
from toolbox.modules import BaseModule
from toolbox.utils.strings import normalize_string, title_case


class LightDialoguePDM(BaseModule):
    '''Persona Dialogue Module based on the LIGHT dataset.'''

    def generator(self) -> t.Generator[list[str], None, None]:
        for episode in LightDialogueDataset():
            # TODO(11b): Scenario doesn't belong in a persona dialog module.
            context_message = f"Scenario: {episode.context[0]}\n"

            persona_message = ""
            for agent in episode.agents:
                persona_message += f"{PromptConstants.pdm_prefix_for(title_case(agent.name))}: {agent.persona}\n"

            episode_messages: t.List[str] = [context_message, persona_message]
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

                # Finally, prepend the turn character's name.
                message = f"{character}: {message}"

                episode_messages.append(message)

            yield episode_messages
