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
from ..datasets import MessageAndRole, TeatimeDataset
from ..utils import PromptManager, fix_style_and_encoding_issues

LOG = logging.getLogger(__name__)

ALL_OOC_PATTERN = re.compile(r"^\[.*\]$")
ASSISTANT_OR_NOTE = r"(?:Assistant|Note)"
HUMAN_OR_STEP = r"(?:Human|S\d{1,})"
KEEP_DIRECTIVE_PATTERN = re.compile(rf"({HUMAN_OR_STEP}: \[|\[{HUMAN_OR_STEP}: )\1(.*?)\2(]\n*)\3")
REMOVE_DIRECTIVE_PATTERN = re.compile(rf"(?:{ASSISTANT_OR_NOTE}: \[|\[{ASSISTANT_OR_NOTE}: ).*?\]\n*")

class TeatimeRoleplayTask(BaseTask):
    '''
    Task to continue a roleplay.
    '''
    def __init__(
        self,
        filters: list[BaseFilter],
        allowed_models: list[str],
        custom_prompts: Optional[list[str]] = None,
        **kwargs
    ) -> None:
        super().__init__(filters=filters)
        # Hold off on establishing the PromptManager because every chat has
        # at least one unique system prompt.
        self.custom_prompts = custom_prompts
        self.allowed_models = allowed_models
        # Set up counters from each file extracted from the Teatime dataset.
        self.file_counters: dict[str, int] = {}

    def __iter__(self) -> Generator[Episode, None, None]:
        for chat in TeatimeDataset():
            # If a chat isn't in the "allowed models", we skip it.
            if chat.model not in self.allowed_models:
                LOG.debug(f"Skipping a chat in {chat.extracted_from} with chat model {chat.model} because it is not in the allowed models.")
                continue

            # Get rid of messages that don't meet certain criteria.
            # But first, we detect if two system prompts are in a row and the
            # second one is not an example chat separator.
            if chat.messages[0].role == "system" and chat.messages[1].role == "system" \
                and chat.messages[1].message != "[Start a new chat]":
                combined_message = chat.messages[0].message.rstrip() + "\n" \
                    + chat.messages[1].message.lstrip()
                
                sys_message = MessageAndRole(
                    message=combined_message,
                    role="system"
                )
                chat.messages[0] = sys_message
                del chat.messages[1]

            # I shit you not, a few of the logs have *NaN* as their content.
            # I've checked where this happened and every log that does this
            # has NaN at the very end of the chat along with a nonsensical user turn.
            # Just delete both. How the fuck does this even happen?
            has_nan = False
            if type(chat.messages[-1].message) == float:
                new_messages = chat.messages[:-2]
                has_nan = True
            
            if not has_nan:
                new_messages = [chat.messages[0]] + \
                    [m for m in chat.messages[1:] if not _skip_criteria(m)]
            else:
                new_messages = [m for m in new_messages if not _skip_criteria(m)]

            # If custom system prompts aren't provided, modify the original
            # system prompt.
            if self.custom_prompts is None:
                #print("Triggering")
                # Now is the time to detect example chats. This is a fucking doozy.
                # First we bring together the entire chat into one string so
                # we can treat this as string splitting.
                only_messages = [m.message for m in new_messages]
                new_chat = "[Start a new chat]"
                # Sometimes new_chat is just in the system prompt for some reason
                # Delete those instances.
                if new_chat in only_messages[0]:
                    only_messages[0] = only_messages[0].replace(new_chat, "")
                    new_messages[0].message = only_messages[0]
                
                entire_chat = "\n".join(only_messages)
                # Check if any example chats even exist. If not, leave it be.
                if new_chat in entire_chat:
                    # Scrub the example chats entirely.
                    # NOTE(TG): Maybe integrate them in later?
                    first_new_chat = only_messages.index(new_chat)
                    last_new_chat = len(only_messages) - 1 - only_messages[::-1].index(new_chat)

                    new_messages = new_messages[:first_new_chat] + new_messages[last_new_chat+1:]
                    system_prompt = new_messages[0].message.replace(new_chat, "").strip()
                    
                # For some reason, a few logs have "{{user}}" rendered as {{user}.
                system_prompt = system_prompt.replace("{{user}", "{{user}}")
                # Now we deal with brackets in the system prompt. Clean this stuff up.
                system_prompt = KEEP_DIRECTIVE_PATTERN.sub(r"\2", system_prompt)
                system_prompt = REMOVE_DIRECTIVE_PATTERN.sub("", system_prompt)
            else:
                system_prompt = PromptManager(custom_prompts=self.custom_prompts).sample_prompt()

            # If there are fewer than 3 messages left at this point, knock it out.
            # Same if there are fewer than 3 unique roles.
            if len(new_messages) < 3 or len(set([r.role for r in new_messages])) < 3:
                continue

            # Set up turns.
            turns: list[Turn] = [
                Turn(
                    utterance=system_prompt,
                    kind=TurnKind.SYSTEM
                ),
            ]
            #[1:] because we already added the first message/system prompt.
            for i, message in enumerate(new_messages[1:]):
                assert message.role != "system", f"Leftover system messages in chat.\nNew messages: {i} | {message}"

                # A few weird double spaces.
                utterance = message.message.replace("  ", " ").replace(r"\r", "")
                utterance = fix_style_and_encoding_issues(utterance)
                turn = Turn(
                    utterance=utterance,
                    kind=TurnKind.MODEL if message.role == "assistant" else TurnKind.USER,
                    name="TODO" if message.role == "assistant" else "You"
                )
                turns.append(turn)

            # Add to the counter.
            if chat.extracted_from not in self.file_counters:
                self.file_counters[chat.extracted_from] = 1
            else:
                self.file_counters[chat.extracted_from] += 1

            chat_identifier = f"teatime-{chat.extracted_from}-{self.file_counters[chat.extracted_from]}"

            episode = Episode(
                turns=turns,
                identifier=chat_identifier
            )
            if self.should_keep(episode):
                # Passed through filters!
                yield episode

def _skip_criteria(message: MessageAndRole) -> bool:
    '''
    Attempts to determine whether a message should be skipped following certain
    criteria.
    '''
    if type(message.message) != str:
        print("wtf", message.message)
    content = message.message.strip()
    # Skip system messages, except for those delineating example chats.
    if message.role == "system" and message.message != "[Start a new chat]":
        return True
    # Skip messages that are empty.
    if len(content) == 0:
        return True
    # RegEx is costly; first scan whether a line begins and ends with brackets.
    if (content[0] != "[" and content[-1] != "]") and message.role != "system":
        if ALL_OOC_PATTERN.match(content):
            return True
    
    return False