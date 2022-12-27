import json
import logging
import os
import re
import typing as t
from dataclasses import dataclass
from waifu.core.consts import PromptConstants

from waifu.datasets import BaseDataset
from waifu.utils.dataset import get_data_path

# The regex used to find message variants (e.g.: `%{Hi|Hello} there!`)
KAJIWOTO_VARIANT_REGEX = re.compile(r'%{(.+?)}')

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KajiwotoMessageResponsePair:
    message_id: str
    bot_id: str

    user_message: str
    bot_response: str
    condition: str


@dataclass(frozen=True)
class BotMetadata:
    bot_id: str
    name: str
    description: str
    personalities: t.List[t.List[str]]
    has_nsfw: bool
    tags: t.List[str]


class KajiwotoDataset(BaseDataset[t.List[KajiwotoMessageResponsePair]]):
    '''
    The Kajiwoto dataset.

    Takes care of properly handling chat history/message context.
    '''

    def __init__(self) -> None:
        self.filepaths = _enumerate_kajiwoto_json_files()
        self.cached_metadata: dict[str, BotMetadata] = {}

    def get_metadata_for_bot(self, bot_id: str) -> BotMetadata:
        '''Returns known medatada for the given bot ID.'''
        if bot_id in self.cached_metadata:
            return self.cached_metadata[bot_id]

        dataset_path = get_data_path(dataset_name="kajiwoto")
        metadata_filepath = os.path.join(dataset_path,
                                         f"{bot_id}_metadata.json")

        with open(metadata_filepath, "r", encoding="utf-8") as metadata_file:
            metadata_dict = json.loads(
                metadata_file.read())["data"]["aiTrainerGroup"]
            metadata = _metadata_dict_to_dataclass(metadata_dict)
            return metadata

    def generator(
            self
    ) -> t.Generator[t.List[KajiwotoMessageResponsePair], None, None]:
        for filepath in self.filepaths:
            with open(filepath, "r", encoding="utf-8") as file:
                messages = json.loads(file.read())["data"]["aiTrainedList"]

                # So, there's a tricky thing to handle in these datasets which
                # is the fact that follow-up messages are saved as completely
                # separate entries in the messages array. For example, if we
                # have a chat log like:
                #
                # Human: 1
                # Bot: 2
                # Human: 3
                # Bot: 4
                #
                # We will have, in the messages array, something like:
                # [
                #   {"userMessage": "3", message: "4", "history": ["1"]},
                #   {"userMessage": "1", message: "2"},
                # ]
                #
                # As far as I could tell, whenever a message has a "history"
                # field, it usually doesn't make sense by itself. Or even by
                # appending history. One needs to look up the original message
                # and reply pair using the history field, then build up the
                # sequence again manually.
                #
                # As such, for each file, we need to load the entire thing into
                # memory to run over it and build an index to do just that
                # (lookups via the history field), so here we go:
                history_contents_to_original_msg_idx: dict[str, int] = {}
                used_message_indexes: t.Set[int] = set()

                for idx, msg in enumerate(messages):
                    if msg["history"]:
                        # Message already references an earlier message-reply
                        # pair. As far as I could tell, that means _this_
                        # specific message can't be referenced, so no point in
                        # saving an index for it here.
                        continue

                    history_contents_to_original_msg_idx[
                        msg["userMessage"]] = idx

                # Now that we have the history index, let's go over _only_ the
                # messages that need to be concatenated with their history.
                for idx, msg in enumerate(messages):
                    if not msg.get("history", None):
                        continue
                    history_contents = msg["history"][0]

                    # Sometimes, a message seems to reference a previous one
                    # that does not exist. Don't know what's up with that, so
                    # let's just ignore.
                    if not history_contents in history_contents_to_original_msg_idx:
                        continue

                    # Fetch the original "history" message to use as context.
                    original_msg_idx = history_contents_to_original_msg_idx[
                        history_contents]
                    original_msg = messages[original_msg_idx]

                    # Yield the conversation episode.
                    yield [
                        _dict_to_dataclass(original_msg),
                        _dict_to_dataclass(msg),
                    ]

                    # Save the indexes of both of these so we don't re-use them
                    # without the proper context.
                    used_message_indexes.add(idx)
                    used_message_indexes.add(original_msg_idx)

                # Now let's go over regular, history-free messages.
                for idx, msg in enumerate(messages):
                    if idx in used_message_indexes:
                        continue

                    yield [_dict_to_dataclass(msg)]


#
# Public helpers.
#

seen_special_tokens: set[str] = set()
seen_scenes: set[str] = set()


def replace_special_tokens_in(string: str) -> str:
    '''
    Replaces known special tokens (e.g.: `%{name}`) with their expected
    equivalents.
    '''
    string = string.replace("%{name}", PromptConstants.USER_TOKEN)
    string = string.replace("%{kajiname}", PromptConstants.BOT_TOKEN)

    if (match := re.search(KAJIWOTO_VARIANT_REGEX, string)) is not None:
        special_token = match.groups()[0]
        if '|' not in special_token and special_token not in seen_special_tokens:
            logger.warning("Unhandled Kajiwoto token: %s", special_token)
            seen_special_tokens.add(special_token)

    if (scene_match := re.search(r"#scene=(.+?)\b", string)) is not None:
        seen_scene = scene_match.groups()[0]
        if seen_scene not in seen_scenes:
            logger.debug("Unhandled Kajiwoto scene: %s", seen_scene)
            seen_scenes.add(seen_scene)

        # Drop the scene marker. Maybe we can use it for something useful, but
        # I can't think of anything at the moment.
        string = string.replace(f"#scene={seen_scene}", "").strip()

    # TODO: There's a few of these which I haven't handled yet. E.g.:
    # %{pronoun} (before and after a dot, so careful with caps).
    return string


def generate_variants_for(
        string: str,
        max_generations: int = 16,
        start_counter_at: int = 0) -> t.Generator[str, None, None]:
    '''
    Given a string like "%{Hello|Hi} there{.|!}, this should yield:

    - Hello there.
    - Hello there!
    - Hi there.
    - Hi there!
    '''

    # Some bot creators went wild with the variants, which causes ridiculous
    # generations if we try to exhaust all possibilities so we cap that here.
    # `start_counter_at` is used for keeping track across recursive calls.
    counter = start_counter_at

    if (match := re.search(KAJIWOTO_VARIANT_REGEX, string)) is not None:
        # Once we have a "%{X|Y|Z}" matched inside the original string, we:
        # - Fetch .groups()[0] (which will give us `X|Y|Z`)
        # - Split by `|` (so we have ["X", "Y", "Z"])
        # - Filter out empty strings
        alternatives = filter(lambda x: x.strip(), match.groups()[0].split("|"))

        # Then, we break the string apart into what comes before and after the
        # alternatives, that way we can re-build with "prefix + choice + sufix".
        prefix = string[:match.start()]
        sufix = string[match.end():]

        for alternative in alternatives:
            variant = f'{prefix}{alternative}{sufix}'

            # However, some strings have multiple variant blocks. In that case,
            # we operate on them recursively until we have just regular strings
            # after generating all possible variants.
            still_have_match = re.search(KAJIWOTO_VARIANT_REGEX,
                                         variant) is not None
            if still_have_match:
                for inner_variant in generate_variants_for(
                        variant, start_counter_at=counter):
                    yield inner_variant

                    # Keep track and break after `max_generations`.
                    counter += 1
                    if max_generations is not None and counter >= max_generations:
                        break
            else:
                yield variant

                # Keep track and break after `max_generations`.
                counter += 1
                if max_generations is not None and counter >= max_generations:
                    break
    else:
        yield string


#
# Private helpers.
#


def _enumerate_kajiwoto_json_files() -> list[str]:
    '''
    Returns a list of paths to all available `.json` files for the `kajiwoto`
    dataset.
    '''
    dataset_path = get_data_path(dataset_name="kajiwoto")
    items = os.listdir(dataset_path)
    files: list[str] = []

    for item in items:
        if not item.endswith(".json"):
            # Don't care about other file types.
            continue

        if item.endswith("_metadata.json"):
            # Don't want to list metadata files here.
            continue

        item_path = os.path.join(dataset_path, item)
        if not os.path.isfile(item_path):
            # Don't care about folders.
            continue

        absolute_item_path = os.path.abspath(os.path.join(dataset_path, item))
        files.append(absolute_item_path)
    return files


def _dict_to_dataclass(obj: dict[str, str]) -> KajiwotoMessageResponsePair:
    return KajiwotoMessageResponsePair(
        message_id=obj["id"],
        bot_id=obj["aiTrainerGroupId"],
        condition=obj["condition"],
        user_message=obj["userMessage"],
        bot_response=obj["message"],
    )


def _metadata_dict_to_dataclass(obj: dict[str, t.Any]) -> BotMetadata:
    return BotMetadata(
        bot_id=obj["id"],
        name=obj["name"],
        description=obj["description"],
        personalities=obj["personalities"],
        has_nsfw=obj["nsfw"],
        tags=obj["tags"],
    )
