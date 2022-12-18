import typing as t

from waifu.datasets.light_dialogue import LightDialogueDataset
from waifu.modules import BaseModule
from waifu.utils.strings import normalize_string, title_case


class LightDialogueVDM(BaseModule):
    '''Vanilla Dialogue Module based on the LIGHT dialogue dataset.'''

    def generator(self) -> t.Generator[str, None, None]:
        for episode in LightDialogueDataset():
            # TODO(11b): Context and persona don't belong in a vanilla dialogue
            # module.
            context_message = f"Context: {episode.context[0]}\n"

            persona_message = ""
            for agent in episode.agents:
                persona_message += f"{title_case(agent.name)}'s Description: {agent.persona}\n"

            episode_messages: t.List[str] = [context_message, persona_message]
            turn_count = len(episode.speech)

            for idx in range(turn_count):
                character = title_case(episode.character[idx])
                speech = normalize_string(episode.speech[idx])

                # Start off with just the actual speech dialogue.
                message = speech

                # If there was an action performed in that turn, add it to the
                # string.
                action = episode.action[idx]
                if action is not None:
                    message += f" *{action}*"

                # If there was an emote in that turn, add it to the string.
                emote = episode.emote[idx]
                if emote is not None:
                    message = f"*{emote}* {message}"

                # Finally, prepend the turn character's name.
                message = f"{character}: {message}"

                episode_messages.append(message)

            yield "\n".join(episode_messages)
