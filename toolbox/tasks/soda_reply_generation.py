import logging
import random
import typing as t

from toolbox.core.models import Episode, Turn, TurnKind
from toolbox.core.task import BaseTask
from toolbox.datasets.soda import SodaDataset
from toolbox.utils.prompts import generate_prompts

LOG = logging.getLogger(__name__)


class SodaReplyGenerationTask(BaseTask):
    '''
    Task to generate a single reply based on given conversation history and
    narrative. Based on SODA data.
    NOTE(TG): Likely requires updating.
    '''

    def __init__(self, split: str = "train") -> None:
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

                # Original model experiments were very sensitive to participant
                # order, so let's randomize to hopefully fix that.
                random.shuffle(participants)

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
            "This reply should be less than 16 words",
            "16 or less words in the message",
            "Have this reply be really short",
            "Short response"
        ])
    elif word_count < 32:
        return random.choice([
            "The generated reply should be of medium length (between 16 to 32 words)",
            "The generated response should be slightly lengthy (at most 32 words)",
            "The generated message should be on the medium side",
            "This message should be between 16 and 32 words in length",
            "Medium response",
            "Reply should be slightly lengthy (16-32 words)"
        ])
    elif word_count < 64:
        return random.choice([
            "The new message will be of moderate-to-large length",
            "The reply should be moderately-sized, tending towards a longer message (more than 32 words)",
            "The generation should be of medium to medium-long length",
            "There should be 32 to 64 words in this reply",
            "The generated message should be somewhere in-between 'medium' and 'long' in terms of length",
            "The range of the number of words in the message should be between 32 and 64."
        ])
    else:
        return random.choice([
            "The new message will be lengthy",
            "The reply should be long, more than 64 words",
            "The generation should be long",
            "This response will be quite lengthy",
            "More than 64 words in the reply, please",
            "The generated message should be more than sixty-four words in length",
            "Very long response (64+ words)"
        ])

_BASE_SYSTEM_PROMPTS = [
    """%{The following is a|Given the following} conversation between {{participants}}:
    
{{conversation}}
    
You %{must complete the conversation by generating a single response|shall generate a response for} {{respond_for}} while adhering to the following %{narrative|summary}:
    
{{narrative}}
    
{{response_length_str}}.""",
    #
    """%{Given the|Pay attention to|Take a look at} the following conversation between {{participants}}:
    
{{conversation}}
    
You %{must|shall|have to} %{generate|create|say|craft} a %{reply|response} for {{respond_for}}, keeping in mind that the conversation must progress according to the following %{summary|synopsis|context}:
    
{{narrative}}
    
The response should be exclusively of human dialogue and contain no roleplaying actions. Replies %{must be|should be no more than} a single paragraph %{long|in length}.""",
    #
    """%{Enter|Engage|Begin|Consider} %{conversation|conversational|chat|quick chat} mode. In this mode, you must %{generate|create} conversational dialogue responses and coherently continue the conversation in %{an interesting|a creative} manner. {{response_length_str}}.
This is the conversation so far:
{{conversation}}
    
These are the themes that the conversation should follow:
{{narrative}}""",
    #
    """%{Consider|Look at|Pay attention to} the following narrative:
    
{{narrative}}
    
You are to generate a response acting as {{respond_for}} in the following conversation between {{participants}}:
    
{{conversation}}
    
{{response_length_str}}.""",
    #
    """Keeping this scenario in mind:
    
{{narrative}}
    
%{Act as|Imitate|Take the role of} {{respond_for}} in this %{chat|conversation} between {{participants}} and reply with a chat message:
    
{{conversation}}
    
Response length: {{response_length_str}}.""",#
    #
    """{{narrative}}
    
Pretend to be {{respond_for}} %{and reply|when replying|as you respond} to the following dialogue history:
{{conversation}}
{{response_length_str}}."""

]

SYSTEM_PROMPTS = generate_prompts(_BASE_SYSTEM_PROMPTS)
