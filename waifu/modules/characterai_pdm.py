import logging
import re
import typing as t

from waifu.core.consts import PromptConstants
from waifu.datasets.characterai import CharacterAiDataset
from waifu.modules import BaseModule

logger = logging.getLogger(__name__)

# Discard episodes shorter than 3 turns. These are likely not very useful for
# the model to learn to converse properly, since they only really contain one
# dialogue response (the first turn is the hardcoded greeting, and the second is
# the user's input).
MIN_EPISODE_LEN = 3

#
# So here's a quick rundown of what needs to happen. We have a limited context
# window (of 2048 tokens, ATM) and for the Persona Dialogue Module (PDM), we
# need to fit all of the following things in there:
#
# - The bot's description/definitions/persona/whatever you want to call it
# - Last X messages of chat history/context (the more the merrier, usually)
# - The user's input message, e.g. `You: [user text here]`
# - The bot's response, e.g. `[Bot name]: [space for the bot's response]`
#
# As such, most of the code here is about taking globs of text and
# chunking/splitting them up to make the format described above fit into blocks
# of 2048-ish tokens (not exactly 2048 because the tokenizer depends on the
# model used, and I don't want to create a dependency on a specific model at the
# data processing stage at this point).
#


class CharacterAiPDM(BaseModule):
    '''A Persona Dialogue Module powered by CharacterAI data.'''

    def generator(self) -> t.Generator[str, None, None]:
        for chat in CharacterAiDataset():
            if len(chat.messages) < MIN_EPISODE_LEN:
                logger.debug(
                    "Found episode shorter than minimum length (%s < %s), discarding.",
                    len(chat.messages), MIN_EPISODE_LEN)
                continue

            base_turns = []
            if chat.bot.description is not None:
                pdm_prefix = PromptConstants.pdm_prefix_for(chat.bot.name)
                pdm_string = f"{pdm_prefix}: {chat.bot.description}"
                base_turns.append(pdm_string)

            if chat.bot.definitions is not None:
                parsed_definitions, parsed_examples = _parse_definitions_for(
                    chat.bot.name, chat.bot.definitions)
                base_turns.append(parsed_definitions)

            # Add an empty turn to separate persona info from messages, if
            # necessary.
            if len(base_turns) > 0:
                base_turns.append("")

            # Now, start adding messages and break episodes apart if they get
            # too big.
            turns = base_turns.copy()
            for idx, raw_message in enumerate(chat.messages):
                # First message is always the bot (since it must send a
                # greeting), and next up is always the user.
                if idx % 2 == 0:
                    # TODO(11b): Handle `[NAME_IN_MESSAGE_REDACTED]`.
                    message = f"{chat.bot.name}: {raw_message}"
                else:
                    message = f"{PromptConstants.USER_PREFIX}: {raw_message}"
                turns.append(message)

                # Splitting logic.
                cur_episode_len = sum([len(x.split()) for x in turns])
                if cur_episode_len > PromptConstants.TARGET_WORD_COUNT_PER_EPISODE:
                    logger.debug(
                        "Episode length went over TARGET_WORD_COUNT_PER_EPISODE, breaking apart."
                    )

                    # Adding the last message made the episode go over the
                    # target word count, so we return the episode without it...
                    removed_turn = turns.pop()
                    yield "\n".join(turns)

                    # ...and start the next episode with the message we had to
                    # trim out from this one.
                    turns = base_turns.copy()
                    turns.append(removed_turn)


#
# Private helpers.
#

EXAMPLE_CHAT_REGEX = re.compile(
    r"({{char}}|{{random_user_\d}}): (.+?)(?:END_OF_DIALOG)", re.DOTALL)
RELAXED_EXAMPLE_CHAT_REGEX = re.compile(r"{{char}}: .+", re.DOTALL)


def _parse_definitions_for(bot_name: str,
                           raw_definitions: str) -> t.Tuple[str, list[str]]:
    '''
    Parses bot definitions.

    This function attempts to find example messages within the input string,
    parses them accordingly and returns them separately from the rest of the
    text in the original `definitions` string.
    '''
    definitions, examples = _parse_definitions_strict(raw_definitions)
    if len(examples) == 0:
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
