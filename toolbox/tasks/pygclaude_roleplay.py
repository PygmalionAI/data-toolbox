import logging
import random

from typing import Generator, Optional

from ..core import (
    BaseFilter,
    BaseTask,
    Episode,
    Turn,
    TurnKind
)
from ..datasets import PygClaudeRpDataset
from ..utils import PromptManager, fix_style_and_encoding_issues

LOG = logging.getLogger(__name__)

class PygClaudeRoleplayTask(BaseTask):
    '''Roleplaying task for the AI Town dataset.'''
    def __init__(
        self,
        filters: list[BaseFilter],
        custom_prompts: Optional[list[str]] = None,
        **kwargs
    ) -> None:
        super().__init__(filters=filters)
        # If no custom prompts, use the generic "assistant" prompts
        self.custom_prompts = custom_prompts
        if custom_prompts is None:
            kwargs = {"custom_prompts": SYSTEM_PROMPTS} if custom_prompts is None \
            else {"custom_prompts": custom_prompts}
        self.prompts = PromptManager(**kwargs)

    def __iter__(self) -> Generator[Episode, None, None]:
        LOG.info("Processing data for task 'PygClaudeRoleplayTask'.")
        for idx, example in enumerate(PygClaudeRpDataset()):
            # Skip if there are less than two messages.
            if len(example.messages) < 2 or len(set([msg.is_human \
                for msg in example.messages])) < 2:
                LOG.debug(f"Skipping conversation {idx} because it has either less than two messages or has only one speaker.")
                continue

            # Construct the system prompt.
            system_prompt = self.prompts.sample_prompt()
            if example.persona is not None:
                system_prompt += f"\n{random.choice(PERSONA_PROMPTS)} " + example.persona

            system_prompt = system_prompt.replace("{{char}}", example.bot_name)
            # If the name is simply "You", we make the user generic
            user_name = example.user_name if example.user_name.lower().strip() \
                != "you" else "the user"
            system_prompt = system_prompt.replace("{{user}}", user_name)

            # Set up turns.
            turns: list[Turn] = [
                Turn(
                    utterance=system_prompt,
                    kind=TurnKind.SYSTEM,
                )
            ]
            # And now just append.
            for message in example.messages:
                utterance = fix_style_and_encoding_issues(message.message).strip()
                kind = TurnKind.USER if message.is_human else TurnKind.MODEL
                turns.append(
                    Turn(
                        utterance=utterance,
                        kind=kind,
                    )
                )

            # Update the system prompt by filling in the template strings.
            turns = self.fill_response_template_strs(turns)
            # Run through the filters.
            episode = Episode(turns=turns, identifier=f"pygclaude-rp-{idx}")
            if self.should_keep(episode):
                # Passed through filters!
                yield episode

SYSTEM_PROMPTS = [
    """%{Enter|Engage|Consider|Begin} %{roleplay|RP|conversation} mode. %{You are to behave as|Pretend to be|You must act as|Roleplay as} {{char}}. %{You must reply|Reply|Respond} to the user while staying in-character. {{response_length_str}}. {{response_style_str}}""",
    """You are {{char}}. %{You must roleplay|Roleplay|Talk} with the user. {{response_style_str}}. {{response_length_str}}""",
    """Name: {{char}}
%{Objective|Task}: %{RP|Roleplay} with {{user}}. Stay %{in-character|IC} and never talk %{out of character|in OOC text}.
Writing length: {{response_length_str}}
Writing style: {{response_style_str}}""",
    "Help write a fictional %{conversation|roleplay|RP} between {{user}} and {{char}}. {{response_length_str}}",
    "%{Start|Begin} a %{RP session|roleplay game} with {{user}}. Note that you are slightly averse to lewdness and that you tend to %{quickly advance|go fast with} the %{story|RP session}."
]

PERSONA_PROMPTS = [
    "{{char}} must act like this persona:",
    "Behave in this manner when emulating {{char}}:",
    "{{char}}'s Persona:",
    "Maintain this personality for {{char}}:",
    "{{char}} can be described like this:",
    "Description of {{char}}:"
]
