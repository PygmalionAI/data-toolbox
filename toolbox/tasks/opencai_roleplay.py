import logging

from typing import Generator, Optional

from ..core import (
    BaseFilter,
    BaseTask,
    Episode,
    Turn,
    TurnKind
)
from ..datasets import OpenCaiDataset
from ..utils import PromptManager, remove_links, fix_style_and_encoding_issues

LOG = logging.getLogger(__name__)

class OpenCaiRoleplayTask(BaseTask):
    '''
    Task to continue a roleplay conversation.
    '''
    def __init__(
        self,
        filters: list[BaseFilter],
        custom_prompts: Optional[list[str]] = None,
        **kwargs
    ) -> None:
        super().__init__(filters=filters)
        # If no custom prompts, use the generic "assistant" prompts
        self.custom_prompts = custom_prompts

    def __iter__(self) -> Generator[Episode, None, None]:
        LOG.info("Processing data for task 'OpenCaiRoleplayTask'.")
        for idx, example in enumerate(OpenCaiDataset()):
            conversation = example.conversation

            # Remove any instance of this weird error string and blanks from the conversation.
            new_convo = []
            for msg in conversation:
                if msg.message != "```\n\u200b\n```" or msg.message.strip() != "":
                    new_convo.append(msg)
            conversation = new_convo

            if len(conversation) < 2:
                LOG.debug(f"Skipping conversation opencai-{idx} because it has less than two messages.")
                continue

            # While the system prompt is a field in the OpenCAI data, it is always
            # the same boilerplate template. We replace it with our own varied templates
            # here.
            if self.custom_prompts is None:
                # Construct the beginning of the prompt.
                sys_prompt = PromptManager(SYSTEM_PROMPTS).sample_prompt()
                # Add the characters.
                char_string = "\n\n".join([f"{c.name}: {c.description}" for c in example.characters]) 
                sys_prompt += ("\n" + char_string)
            else:
                sys_prompt = PromptManager(self.custom_prompts).sample_prompt()

            turns: list[Turn] = [
                Turn(
                    utterance=sys_prompt,
                    kind=TurnKind.SYSTEM,
                    name="System"
                )
            ]

            # Establish things.
            kind = TurnKind.USER
            full_msg = ""

            # Add the rest of the turns.
            for i, msg in enumerate(conversation):
                text = msg.message
                # Discard messages from the MEE6 bot, since they're just notifications.
                if msg.author == "MEE6":
                    continue

                # Clean the message.
                if text.startswith("{Attachment}"):
                    # Discard.
                    continue
                # Remove links.
                text = remove_links(text)
                # Remove embeds.
                text = text.replace("{Embed}\n", "")
                # Fix up messages.
                text = fix_style_and_encoding_issues(text)
                # More normalization.
                text = text.replace('“', '"').replace('”', '"').replace("’", "'")
                text = text.strip()
                # Now we check to see if the author is the same as the previous one.
                # If so, don't add a new turn, but add to the previous one.
                try:
                    if msg.author != conversation[i-1].author:
                        # Yield this turn with its current kind.
                        if full_msg != "":
                            turns.append(
                                Turn(
                                    utterance=full_msg,
                                    kind=kind,
                                    name=msg.author
                                )
                            )
                        # Reset the full_msg and switch the kind
                        full_msg = text
                        kind = TurnKind.USER if kind == TurnKind.MODEL else TurnKind.MODEL
                    else:
                        full_msg += f" {text}"
                except IndexError:
                    # This is the first message.
                    full_msg = text

            # Now yield the Episode after filtering.
            episode = Episode(turns=turns, identifier=f"opencai-{idx}")
            if self.should_keep(episode):
                # Passed through filters!
                yield episode

SYSTEM_PROMPTS = [
    "%{Enter|Begin|Start in} %{RP|roleplay|role-play|conversational RP|role play} mode. You must adhere to the following %{characters|personas} and their %{descriptions|personalities}.",
    "%{Start|Begin|Commence} a %{roleplay|chat} between the following characters:",
    "You are to %{imitate|roleplay as|act like} %{the following|these} characters. %{Here they are|They are listed as such|The list of characters are}:",
    "list of %{personalities|personas|character descriptions}:",
    "You are in a never-ending fictional %{roleplay|RP}.",
    "%{Personas|Character descriptions|Characters} to %{roleplay as|imitate|act like}:",
]