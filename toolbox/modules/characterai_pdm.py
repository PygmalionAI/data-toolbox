import logging
import re
import typing as t

from toolbox.core.consts import PromptConstants
from toolbox.core.models import Episode, Turn
from toolbox.datasets.characterai import CharacterAiDataset
from toolbox.modules import BaseModule

LOG = logging.getLogger(__name__)

MASK_CHARACTER_NAMES = True


class CharacterAiPDM(BaseModule):
    '''A Persona Dialogue Module powered by CharacterAI data.'''

    def generator(self) -> t.Generator[Episode, None, None]:
        for chat in CharacterAiDataset():
            turns: list[Turn] = []
            for raw_message in chat.messages:
                message_text = _process_message(raw_message.text)
                speaker = PromptConstants.USER_PREFIX \
                    if raw_message.is_human else chat.bot.name

                if MASK_CHARACTER_NAMES:
                    message_text = re.sub(
                        rf"\b{chat.bot.name}\b",
                        PromptConstants.BOT_TOKEN,
                        message_text,
                    )
                    speaker = PromptConstants.USER_PREFIX \
                        if raw_message.is_human else PromptConstants.BOT_TOKEN

                turns.append(
                    Turn(
                        utterance=message_text,
                        speaker=speaker,
                        human_speaker=raw_message.is_human,
                    ))

            personas = {}
            if chat.bot.description is not None:
                personas[PromptConstants.BOT_TOKEN] = chat.bot.description
            if chat.bot.definitions is not None:
                parsed_definitions, parsed_examples = _parse_definitions_for(
                    chat.bot.name, chat.bot.definitions)
                parsed_definitions = _process_definitions(parsed_definitions)
                parsed_examples = [
                    _process_definitions(example) for example in parsed_examples
                ]

                # TODO(11b): Figure out where and how to use these. They're
                # usually big enough that just adding them to the prompt fills
                # up the context window too much.
                #
                # LOG.warning("parsed_definitions: %s", parsed_definitions)
                # LOG.warning("parsed_examples: %s", parsed_examples)

            yield Episode(turns=turns, participant_personas=personas)


#
# Private helpers.
#

EXAMPLE_CHAT_REGEX = re.compile(
    r"({{char}}|{{random_user_\d}}): (.+?)(?:END_OF_DIALOG)", re.DOTALL)
RELAXED_EXAMPLE_CHAT_REGEX = re.compile(r"{{char}}: .+", re.DOTALL)
EXCESSIVE_ELLIPSIS_REGEX = re.compile(r"\.{4,}")


def _process_message(original_string: str) -> str:
    '''
    Processes a single message to clean it up and filter/replace the appropriate
    special tokens.
    '''
    string = EXCESSIVE_ELLIPSIS_REGEX.sub("...", original_string)

    # TODO(11b): Improve this.
    string = string.replace("[NAME_IN_MESSAGE_REDACTED]",
                            PromptConstants.USER_TOKEN)
    string = string.replace("[REDACTED]", PromptConstants.USER_TOKEN)
    string = string.replace("[FIRST_NAME_REDACTED]", PromptConstants.USER_TOKEN)
    string = string.replace("[USERNAME_REDACTED]", PromptConstants.USER_TOKEN)
    string = string.replace("[NAME_REDACTED]", PromptConstants.USER_TOKEN)
    return string.strip()


def _process_definitions(original_string: str) -> str:
    '''Replaces known special tokens for which we have equivalents for.'''
    string = original_string.replace("{{user}}: ", "You: ")
    string = string.replace("{{user}}", PromptConstants.USER_TOKEN)
    string = string.replace("END_OF_DIALOG", PromptConstants.CHAT_START_TOKEN)
    return string


def _parse_definitions_for(bot_name: str,
                           raw_definitions: str) -> t.Tuple[str, list[str]]:
    '''
    Parses bot definitions.

    This function attempts to find example messages within the input string,
    parses them accordingly and returns them separately from the rest of the
    text in the original `definitions` string.
    '''
    definitions, examples = _parse_definitions_strict(raw_definitions)
    if len(examples) == 0 or len(definitions.strip()) == 0:
        definitions, examples = _parse_definitions_relaxed(raw_definitions)

    parsed_definitions = definitions.replace("{{char}}", bot_name)
    parsed_examples = [x.replace("{{char}}", bot_name) for x in examples]

    return parsed_definitions, parsed_examples


def _parse_definitions_strict(definitions: str) -> t.Tuple[str, list[str]]:
    '''
    Strict parsing of a bot's definitions string, assumes END_OF_DIALOG was used
    correctly by the bot's creator.
    '''
    matched_example_chats = EXAMPLE_CHAT_REGEX.finditer(definitions)
    examples = [
        x.group().replace("END_OF_DIALOG", "").strip()
        for x in matched_example_chats
    ]
    definitions_without_examples = re.sub(EXAMPLE_CHAT_REGEX, "", definitions)

    return definitions_without_examples, examples


def _parse_definitions_relaxed(definitions: str) -> t.Tuple[str, list[str]]:
    '''
    Same as the `_parse_definitions_strict`, but this one is much more relaxed
    and should be used for when the bot creator didn't properly use
    END_OF_DIALOG to delineate example chats.
    '''
    matched_example_chats = RELAXED_EXAMPLE_CHAT_REGEX.finditer(definitions)
    examples = [x.group().strip() for x in matched_example_chats]
    definitions_without_examples = re.sub(RELAXED_EXAMPLE_CHAT_REGEX, "",
                                          definitions)

    return definitions_without_examples, examples
