import logging
import random
import re
import typing as t

from toolbox.core.consts import PromptConstants
from toolbox.datasets.characterai import CharacterAiDataset
from toolbox.modules import BaseModule

logger = logging.getLogger(__name__)

# Discard episodes shorter than 3 turns. These are likely not very useful for
# the model to learn to converse properly, since they only really contain one
# dialogue response (the first turn is the hardcoded greeting, and the second is
# the user's input).
MIN_EPISODE_LEN = 3

# Discard episodes where the average similarity between the bot's messages is
# higher than this value.
EPISODE_SIMILARITY_THRESHOLD = 0.55

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

    def generator(self) -> t.Generator[list[str], None, None]:
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

            # Add turn to separate persona info from messages, if
            # necessary.
            if len(base_turns) > 0:
                base_turns.append(PromptConstants.CHAT_START_TOKEN)

            # Now, start adding messages and break episodes apart if they get
            # too big.
            turns = base_turns.copy()
            bot_messages: list[str] = []

            for raw_message in chat.messages:
                message_text = _process_message(raw_message.text)
                if raw_message.is_human:
                    message = f"{PromptConstants.USER_PREFIX}: {message_text}"
                else:
                    message = f"{chat.bot.name}: {message_text}"
                    bot_messages.append(message_text)
                turns.append(message)

                # Splitting logic.
                cur_episode_len = sum([len(x.split()) for x in turns])
                if cur_episode_len > PromptConstants.TARGET_WORD_COUNT_PER_EPISODE:
                    logger.debug(
                        "Episode length went over TARGET_WORD_COUNT_PER_EPISODE (%s > %s), breaking apart.",
                        cur_episode_len,
                        PromptConstants.TARGET_WORD_COUNT_PER_EPISODE)

                    # Drop a % of episodes that have undesirable behaviors.
                    if any([_seems_embarassed(x) for x in bot_messages]):
                        should_skip = random.randint(0, 100) < 70
                        if should_skip:
                            for bot_message in bot_messages:
                                if _seems_embarassed(bot_message):
                                    logger.debug("Skipping over a tomato: `%s`",
                                                 bot_message)
                            turns = base_turns.copy()
                            bot_messages = []
                            continue

                    if any([_has_stuttering(x) for x in bot_messages]):
                        should_skip = random.randint(0, 100) < 50
                        if should_skip:
                            for bot_message in bot_messages:
                                if _has_stuttering(bot_message):
                                    logger.debug(
                                        "Skipping over stuttering: `%s`",
                                        bot_message)
                            turns = base_turns.copy()
                            bot_messages = []
                            continue

                    if len(bot_messages) < 2:
                        turns = base_turns.copy()
                        bot_messages = []
                        continue

                    # Calculate similarity between sequential bot message pairs
                    # within this episode, and drop it if it goes above the
                    # defined threshold.
                    similarity_score_matrix = _calculate_similarity_scores(
                        bot_messages)
                    average_similarity_score_for_episode = 0.0
                    for score in similarity_score_matrix[0]:
                        if score == 1:
                            continue
                        average_similarity_score_for_episode += score
                        average_similarity_score_for_episode /= 2

                    # Adding the last message made the episode go over the
                    # target word count, so we return the episode without it...
                    removed_turn = turns.pop()
                    if average_similarity_score_for_episode <= EPISODE_SIMILARITY_THRESHOLD:
                        yield [_replace_special_tokens(turn) for turn in turns]
                    else:
                        logger.debug(
                            "Ignoring episode due to high similarity between messages (%s > %s)",
                            average_similarity_score_for_episode,
                            EPISODE_SIMILARITY_THRESHOLD)

                    # ...and start the next episode with the message we had to
                    # trim out from this one.
                    turns = base_turns.copy()
                    turns.append(removed_turn)
                    bot_messages = []

            # Episode was too small to break up, return as-is if it has at least
            # a _little_ bit of data.
            if len(turns) > 2:
                # FIXME(11b): This bypasses all the checks above. Refactor so
                # the code is structured properly and shares the checks.
                yield [_replace_special_tokens(turn) for turn in turns]


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
    string = string.replace("[NAME_IN_MESSAGE_REDACTED]",
                            PromptConstants.USER_TOKEN)
    return string.strip()


def _replace_special_tokens(original_string: str) -> str:
    '''Replaces known special tokens for which we have equivalents for.'''
    string = original_string.replace("{{user}}: ", "You: ")
    string = string.replace("{{user}}", "<USER>")
    return string


def _seems_embarassed(utterance: str) -> bool:
    '''Most advanced sentiment classification algorithm ever invented.'''
    last_idx = -1
    asterisk_idxs = []
    while (last_idx := utterance.find("*", last_idx + 1)) != -1:
        asterisk_idxs.append(last_idx)

    while True:
        if len(asterisk_idxs) < 2:
            break

        start_idx = asterisk_idxs.pop(0)
        end_idx = asterisk_idxs.pop(0)
        action = utterance[start_idx:end_idx].lower()

        if any([
                x in action for x in
            ["blush", "like a tomato", "as a tomato", "embarass", "nervous"]
        ]):
            return True
    return False


def _has_stuttering(utterance: str) -> bool:
    '''Tries to detect stuttering like "W-what?".'''
    for word in utterance.split():
        if "-" not in word:
            continue

        if len(word) < 4:
            continue

        if word[1] != "-":
            continue

        lowercased_word = word.lower()
        if lowercased_word[0] == lowercased_word[2]:
            return True

    return False


def _calculate_similarity_scores(bot_turns: list[str]) -> t.Any:
    '''
    Calculates similarity scores between bot turns.

    This is a roundabout way to try and _possibly_ detect the post-1.1 CAI
    looping behavior so we can handle it during the data preprocessing.
    '''
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(bot_turns)
    arr = x.toarray()

    sims = cosine_similarity(arr)
    return sims


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
