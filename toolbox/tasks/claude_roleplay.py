import logging
import random
import typing as t

from toolbox.core.models import Episode, Turn, TurnKind
from toolbox.core.task import BaseTask
from toolbox.datasets.claude_logs import ClaudeRpDataset
from toolbox.utils.prompts import generate_prompts, select_prompt

LOG = logging.getLogger(__name__)

class ClaudeRoleplayTask(BaseTask):
    '''Roleplay task based on Claude logs'''
    def __iter__(self) -> t.Generator[Episode, None, None]:
        for convo in ClaudeRpDataset():
            # Deal with system prompts
            system_prompt = select_prompt(SYSTEM_PROMPTS)
            # Add a persona if there is one
            if convo.persona is not None and system_prompt != "":
                system_prompt += f"\n{random.choice(PERSONA_PROMPTS)} " + convo.persona
            
            system_prompt = system_prompt.replace("{{char}}", convo.bot_name)
            # If the name is simply "You", we make the user generic
            if convo.user_name.lower().strip() != "you":
                system_prompt = system_prompt.replace("{{user}}", convo.user_name)
            else:
                system_prompt = system_prompt.replace("{{user}}", "the user")
                
            turns: list[Turn] = [
                Turn(
                    utterance=system_prompt,
                    kind=TurnKind.SYSTEM,
                )
            ]

            for message in convo.messages:
                turns.append(Turn(
                    utterance=message.message,
                    kind=TurnKind.USER if message.is_user else TurnKind.MODEL
                ))

            # Cut off any logs that don't have one full exchange of conversation
            if len(turns) <= 2:
                LOG.info("Skipping conversation {convo.convo_id} due to insufficient conversation length.")
                continue

            yield Episode(
                turns=turns,
                identifier=f"claude-rp-{convo.convo_id}"
            )

_BASE_SYSTEM_PROMPTS = [
    """%{Enter|Engage|Consider|Begin} %{roleplay|RP|conversation} mode. %{You are to behave as|Pretend to be|You must act as|Roleplay as} {{char}}. %{You must reply|Reply|Respond} to the user while staying in-character. {{response_length_str}}. {{response_style_str}}""",
    """You are {{char}}. %{You must roleplay|Roleplay|Talk} with the user. {{response_style_str}}. {{response_length_str}}""",
    """Name: {{char}}
%{Objective|Task}: %{RP|Roleplay} with {{user}}. Stay %{in-character|IC} and never talk %{out of character|in OOC text}.
Writing length: {{response_length_str}}
Writing style: {{response_style_str}}""",
    "Help write a fictional %{conversation|roleplay|RP} between {{user}} and {{char}}. {{response_length_str}}",
    "%{Start|Begin} a %{RP session|roleplay game} with {{user}}. Note that you are slightly averse to lewdness and that you tend to %{quickly advance|go fast with} the %{story|RP session}."
    "",
]

SYSTEM_PROMPTS = generate_prompts(_BASE_SYSTEM_PROMPTS)

PERSONA_PROMPTS = [
    "{{char}} must act like this persona:",
    "Behave in this manner when emulating {{char}}:",
    "{{char}}'s Persona:",
    "Maintain this personality for {{char}}:",
    "{{char}} can be described like this:"
]
