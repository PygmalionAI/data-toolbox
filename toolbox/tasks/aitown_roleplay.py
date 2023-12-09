import logging
import re

from typing import Generator, Optional

from ..core import (
    BaseFilter,
    BaseTask,
    Episode,
    Turn,
    TurnKind
)
from ..datasets import AiTownDataset
from ..utils import PromptManager

LOG = logging.getLogger(__name__)

INTRODUCTION_PATTERN = re.compile(r"(?:^You(?:'re| are ))(.+),.+ conversation with (.+)\.")

class AiTownRoleplayTask(BaseTask):
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
        self.kind_mapping: dict[str, TurnKind] = {
            "user": TurnKind.USER,
            "assistant": TurnKind.MODEL
        }

    def __iter__(self) -> Generator[Episode, None, None]:
        LOG.info("Processing data for task 'AiTownRoleplayTask'.")
        for idx, example in enumerate(AiTownDataset()):
            conversation = example.conversation

            if len(conversation) < 3:
                LOG.debug(f"Skipping conversation aitown-{idx} because it has less than three messages.")
                continue

            # Process the system prompt first, which is always the first message.
            sys_message = conversation[0].message
            # Grab the names of the two speakers.
            matches = INTRODUCTION_PATTERN.match(sys_message).groups()
            assert len(matches) == 2, f"Names weren't captured properly! Groups: {matches}"
            user_name, bot_name = matches
            # Fix sysprompt.
            sys_message = _fix_spaces(sys_message)

            # Set up turns.
            turns = [
                Turn(
                    utterance=sys_message,
                    kind=TurnKind.SYSTEM,
                    name="System"
                )
            ]

            for message in conversation[1:]:
                cleaned_msg = _fix_spaces(message.message)
                current_kind = self.kind_mapping[message.role]
                turn = Turn(
                    utterance=cleaned_msg,
                    kind=current_kind,
                    name=user_name if current_kind == TurnKind.USER else bot_name
                )
                turns.append(turn)

            # Run through the filters.
            episode = Episode(turns=turns, identifier=f"aitown-{idx}")
            if self.should_keep(episode):
                # Passed through filters!
                yield episode
                
def _fix_spaces(message: str) -> str:
    '''Fix spaces in a message.'''
    return message.replace("\n      ", "\n").strip()
