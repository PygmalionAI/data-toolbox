import logging
import random
import typing as t

from toolbox.core.models import Episode, Turn, TurnKind
from toolbox.core.task import BaseTask
from toolbox.datasets.soda import SodaDataset

LOG = logging.getLogger(__name__)


class SodaReplyGenerationTask(BaseTask):
    '''
    Task to generate a single reply based on given conversation history and
    narrative. Based on SODA data.
    '''

    def __init__(self, split: str) -> None:
        self.split = split

        super().__init__()

    def __iter__(self) -> t.Generator[Episode, None, None]:
        for conversation in SodaDataset(split=self.split):
            cur_history: list[str] = []

            for idx, utterance in enumerate(conversation.dialogue):
                speaker_name = conversation.speakers[idx]
                cur_history.append(f"{speaker_name}: {utterance}")

                if len(cur_history) < 4:
                    # Too little data to build up a decent prompt, let's keep
                    # going.
                    continue

                participants = list(set(conversation.speakers))
                participants_str = " and ".join(
                    [", ".join(participants[:-1]), participants[-1]])

                history_str = "\n".join(cur_history[:-2])
                response_length_str = _response_length_str_for(utterance)

                system_prompt = random.choice(SYSTEM_PROMPTS)
                system_prompt = system_prompt.replace("{{participants}}",
                                                      participants_str)
                system_prompt = system_prompt.replace("{{conversation}}",
                                                      history_str)
                system_prompt = system_prompt.replace("{{narrative}}",
                                                      conversation.narrative)
                system_prompt = system_prompt.replace("{{respond_for}}",
                                                      speaker_name)
                system_prompt = system_prompt.replace("{{response_length_str}}",
                                                      response_length_str)

                system_turn = Turn(system_prompt, TurnKind.SYSTEM)
                # TODO(11b): Add a variant where the speaker's name is omitted
                # randomly, both in the user and the model turns. Adjust the
                # system prompt accordingly.
                user_turn = Turn(cur_history[-2], TurnKind.USER)
                model_turn = Turn(cur_history[-1], TurnKind.MODEL)
                turns = [system_turn, user_turn, model_turn]

                yield Episode(
                    turns,
                    identifier=
                    f"soda-{self.split}-{conversation.original_index}-reply-generation"
                )


def _response_length_str_for(response: str) -> str:
    word_count = len(response.split())

    if word_count < 16:
        return random.choice([
            "The generated response should be short (less than 16 words)",
            "Be brief when generating the message (less than sixteen words)",
            "The generated reply should be small",
        ])
    elif word_count < 32:
        return random.choice([
            "The generated reply should be of medium length (between 16 to 32 words)",
            "The generated response should be slightly lengthy (at most 32 words)",
            "The generated message should be on the medium side",
        ])
    elif word_count < 64:
        return random.choice([
            "The new message will be of moderate-to-large length",
            "The reply should be moderately-sized, tending towards a longer message (more than 32 words)",
            "The generation should be of medium to medium-long length",
        ])
    else:
        return random.choice([
            "The new message will be lengthy",
            "The reply should be long, more than 64 words",
            "The generation should be long",
        ])


SYSTEM_PROMPTS = [
    """The following is a conversation between {{participants}}:

{{conversation}}

You must complete the conversation by generating a single response for {{respond_for}}, while adhering to the following narrative:

{{narrative}}

The generated response must be a single paragraph, and must not contain roleplay or actions enclosed between asterisks. {{response_length_str}}.""",

    #
    #
    #
    """Given the following conversation between {{participants}}:

{{conversation}}

You shall generate a response for {{respond_for}}, keeping in mind that the conversation must progress according to the following summary:

{{narrative}}

{{response_length_str}}. The response should be exclusively of human dialogue and contain no roleplaying actions.""",

    #
    #
    #
    """Enter conversation mode. In this mode, you must generate conversational dialogue responses and coherently continue the conversation in an interesting manner. {{response_length_str}}.

This is the conversation so far:

{{conversation}}

These are the themes that the conversation should follow:

{{narrative}}
""",

    #
    #
    #
    """This is a conversation involving {{participants}}:

{{conversation}}

Generate a message for {{respond_for}}, taking into account that the conversation should follow this scenario:

{{narrative}}

The message should contain sentences that strictly denote spoken dialogue, and not actions. {{response_length_str}}.""",

    #
    #
    #
    """You are to generate a message for {{respond_for}} in this chat between {{participants}}. This is the chat so far:

{{conversation}}

Keep in mind this context:

{{narrative}}

{{response_length_str}}.""",

    #
    #
    #
    """Consider the following narrative:

{{narrative}}

You are to generate a response acting as {{respond_for}} in the following conversation between {{participants}}:

{{conversation}}

{{response_length_str}}.""",

    #
    #
    #
    """Keeping this scenario in mind:

{{narrative}}

Act as {{respond_for}} in this chat between {{participants}} and reply with a chat message:

{{conversation}}

{{response_length_str}}.""",

    #
    #
    #
    """{{narrative}}

Pretend to be {{respond_for}} reply to the following dialogue history:

{{conversation}}

{{response_length_str}}.""",
]
