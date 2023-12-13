import logging

from typing import Generator, Optional

from ..core import (
    BaseFilter,
    BaseTask,
    Episode,
    Turn,
    TurnKind
)
from ..datasets import AesirDataset
from ..utils import PromptManager

LOG = logging.getLogger(__name__)

class AesirRoleplayTask(BaseTask):
    def __init__(
        self,
        filters: list[BaseFilter],
        custom_prompts: Optional[list[str]] = None,
        **kwargs
    ) -> None:
        '''
        Task to roleplay with personas.
        '''
        super().__init__(filters=filters)
        # Hold off on establishing the PromptManager because every chat has
        # at least one unique system prompt.
        self.custom_prompts = custom_prompts
        if self.custom_prompts is not None:
            self.prompts = PromptManager(**kwargs)
        else:
            self.prompts = None # useful as sanity check

        # ShareGPT format.
        self.kind_map: dict[str, TurnKind] = {
            "system": TurnKind.SYSTEM,
            "human": TurnKind.USER,
            "gpt": TurnKind.MODEL
        }

    def __iter__(self) -> Generator[Episode, None, None]:
        LOG.info("Processing data for task 'AesirRoleplayTask'.")
        for i, c in enumerate(AesirDataset()):
            # If there are less than three messages, we skip it.
            if len(c.conversation) < 3:
                LOG.debug(f"Skipping conversation aesir-{i} because it has less than three messages.")
                continue

            turns: list[Turn] = []

            # Handle possible custom system prompts.
            if self.custom_prompts is None:
                conversation = c.conversation
            else:
                assert self.prompts is not None, "self.prompts is None when it shouldn't be!"
                sys_prompt = self.prompts.sample_prompt()
                turns.append(Turn(
                    utterance=sys_prompt,
                    kind=TurnKind.SYSTEM,
                    name="System"
                ))
                conversation = c.conversation[1:]

            # Iterate through the dataset.
            for message in conversation:
                utterance = message.message.strip()
                # All users are named "Anon".
                utterance = utterance.replace("Anon", "{{user}}")
                kind = self.kind_map[message.role]

                # Append to turns.
                turns.append(Turn(
                    utterance=utterance,
                    kind=kind,
                    name="TODO" # literally too tired to do this atm lol
                ))

            # Run through the filters.
            episode = Episode(turns=turns, identifier=f"aesir-{i}")
            if self.should_keep(episode):
                # Passed through filters!
                yield episode
