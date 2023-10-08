import logging
import random
import re
import typing as t

from markdownify import markdownify

from toolbox.core.models import Episode, Turn, TurnKind
from toolbox.core.task import BaseTask
from toolbox.datasets.rp_guild import RpGuildDataset
# No need to re-invent the wheel.
from toolbox.tasks.rp_forums_writing import(
    _fix_markdown,
    _fix_style_and_encoding_issues,
    _not_usable_as_training_label,
    _remove_bad_html_tags,
    _remove_links,
    _remove_trailing_whitespace_and_bad_lines,
    _seems_to_have_ooc_talk,
    _split_message,
)
from toolbox.utils.prompts import generate_prompts, select_prompt

# Gaze upon my works, ye mighty, and despair.
MENTION_PATTERN = re.compile(r"(?<!\w)([^\S\r\n]|^)*@[^\W\s]+?(?=(,|\.|\?|~|!|\s|:|$))", flags=re.MULTILINE)
OOC_PATTERN = re.compile(r"((\[\[|\(\().*(\)\)|\]\])|\(OOC:.+\)|(?<=\s)OOC:.*(?!$))")

LOG = logging.getLogger(__name__)

class RpGuildWritingTask(BaseTask):
    '''
    Task to generate an appropriate continuation in the context of a fantasy roleplay.
    '''
    def __init__(self, all_model_turns: bool = False, keep_ooc: bool = False) -> None:
        # Keep the old way of having the turns be almost entirely model turns
        # just in case.
        self.all_model_turns = all_model_turns
        self.keep_ooc = keep_ooc

    def __iter__(self) -> t.Generator[Episode, None, None]:
        for thread in RpGuildDataset():
            # Eliminate threads which are deemed 'unsalvagable'.
            if thread.thread_name in BROKEN_THREADS:
                continue

            # Skip over OOC/character threads
            # TODO(TG): If I have time, I might try doing a very complex thing where I can fetch definitions
            # from char threads, but I think it's too much work for now.
            if thread.thread_type != "IC":
                continue

            # Prune threads with less than 2 messages
            if len(thread.messages) < 2:
                LOG.debug(f"Skipping {thread.thread_name} with only one message")
                continue

            # Build up a dictionary of usernames to replace for privacy reasons.
            usernames = set([message.author for message in thread.messages])
            username_substitutions: dict[str, str] = {}
            for idx, name in enumerate(usernames):
                username_substitutions[name] = "{{char_" + str(idx) + "}}"

            # NOTE(TG): For now, I'm having this be 1x1 roleplays only, but I really do
            # want this to account for group roleplays. I'll figure something out later.
            if len(usernames) > 2 and "1x1" not in thread.tags:
                continue

            # Generate the system prompt.
            sys_prompt = select_prompt(SYSTEM_PROMPTS)
            # Takes the first style prompt it sees
            for tag in thread.tags:
                if tag in list(STYLE_PROMPT_MAPPING.keys()):
                    sys_prompt += select_prompt(STYLE_PROMPT_MAPPING[tag])
                    break
            # The time and genre
            genre_str, time_str = _combine_tags_into_str(thread.tags)
            if genre_str is not None:
                add_prompt = select_prompt(GENRE_PROMPTS)
                sys_prompt += (add_prompt + genre_str + ".")
            if time_str is not None:
                add_prompt = select_prompt(TIME_PROMPTS)
                sys_prompt += (add_prompt + time_str + ".")
            # NSFW
            if "18+" in thread.tags:
                sys_prompt += select_prompt(NSFW_PROMPTS)

            # Finally convert the system prompt to a Turn
            sys_prompt = Turn(utterance=sys_prompt, kind=TurnKind.SYSTEM)
            turns: list[Turn] = [sys_prompt]

            # Since CAI-like UIs can have the model speak first,
            # we augment the data by allowing the model to sometimes
            # speak first. Specifically, only 25% of the time.
            # This is only used when all_model_turns is False.
            current_speaker = random.choice([TurnKind.MODEL, TurnKind.USER, TurnKind.USER, TurnKind.USER])
            
            for message in thread.messages:
                long_message = message.message

                long_message = _fix_style_and_encoding_issues(long_message)
                long_message = _remove_bad_html_tags(long_message)
                long_message = _remove_links(long_message)

                assert "http://" not in long_message and "https://" not in long_message \
                    , "Failed to clean URLs properly."

                # Add some variety so we can generate a synthetic prompt for
                # controlling generation length down the line.
                target_word_count = random.randint(200, 600)

                for message in _split_message(
                        long_message,
                        target_word_count=target_word_count,
                        delimiter="<br/><br/>"):
                    cleaned_message = str(markdownify(message))
                    cleaned_message = _remove_trailing_whitespace_and_bad_lines(
                        cleaned_message)

                    cleaned_message = _fix_markdown(cleaned_message)

                    # Fix excessive spaces after converting to Markdown.
                    cleaned_message = re.sub("\n{2,}", "\n\n", cleaned_message)

                    # Username substitutions need to be done _after_ the HTML has
                    # been converted into markdown, otherwise we get escape
                    # characters messing things up.
                    for name, substitution in username_substitutions.items():
                        cleaned_message = re.sub(rf"\b{re.escape(name)}\b",
                                                 substitution, cleaned_message)
                        
                    # Now remove mentions and clean OOC as well if specified
                    if not self.keep_ooc:
                        cleaned_message = _remove_ooc(cleaned_message)
                    cleaned_message = _remove_mentions(cleaned_message)
                        
                    # NOTE(TG): See note in rp_forums_writing.py for explanation
                    # on why we don't have RP data be all model turns anymore.
                    if self.all_model_turns:
                        # Little bit of roundabout logic so here's some explanation
                        # as we go. We start by marking everything as a model turn
                        # so we use as much data as possible as training labels.
                        turn_kind = TurnKind.MODEL
                        if _not_usable_as_training_label(cleaned_message):
                            # ...however, if we have some problem in the data that
                            # we'd rather not see the model replicate, we mark it
                            # as a human turn, which is used as context but not for
                            # loss calculation during training.
                            turn_kind = TurnKind.USER
                        elif _seems_to_have_ooc_talk(cleaned_message) \
                            and not _seems_to_have_ooc_talk(turns[-1].utterance):
                            # _However_, there's also another case we'd like to
                            # handle. Ideally, the model should not slip into OOC
                            # talk unprompted - it should only do that if we've
                            # tried to talk to it out-of-character first.
                            #
                            # So if this turn has OOC talk, we'll only use it as a
                            # model turn if the previous (user) turn also had OOC
                            # talk.
                            turn_kind = TurnKind.USER
                    else:
                        # TODO(TG): Try to do more about OOC/potential low-quality generations.
                        turn_kind = current_speaker

                    # If the message is blank for whatever reason, discard
                    cleaned_message = cleaned_message.strip()
                    if cleaned_message == "":
                        continue

                    turn = Turn(utterance=cleaned_message, kind=turn_kind)
                    turns.append(turn)

            # Messy switching
            current_speaker = TurnKind.MODEL if current_speaker == TurnKind.USER \
                else TurnKind.USER
            
            yield Episode(
                turns=turns,
                identifier=f"rp-guild-{thread.thread_name}",
            )


def _remove_mentions(message: str) -> str:
    '''Removes username mentions from the message.'''
    cleaned_message = message
    removal_bounds: list[tuple[int, int]] = []
    for match in re.finditer(MENTION_PATTERN, message):
        end_char = message[match.end()-1]
        # If the next character is a whitespace or punctuation,
        # we can assume that removing the mention won't affect the message
        # much in terms of coherency. We store the bounds in a tuple
        # so we can take out all the mentions at once later.
        if end_char in [" ", ".", "!", "?"]:
            removal_bounds.append(match.span())
        # Else, we leave it be.

    # Clean the message now.
    if len(removal_bounds) > 0:
        # Set up an offset for adjusting the position of the next bounds
        # after the mention is deleted.
        offset = 0
        for start, end in removal_bounds:
            start -= offset
            end -= offset
            offset += end - start
            cleaned_message = cleaned_message[:start] + cleaned_message[end:]

    # There's sometimes weirdness where at the beginning, a whitespace character can remain.
    # Impromptu patch that here.
    return cleaned_message.strip()

def _remove_ooc(message: str) -> str:
    return re.sub(OOC_PATTERN, "", message)

# An absolute nightmare of constants and prompt generations.

def _combine_tags_into_str(tags: list) -> tuple[str, str]:
    '''Combines tags into a string.'''
    def construct_conjunction(tags: list) -> str:
        '''
        Converts lists of tags into a natural sounding sentence. Works like this:
        Given no tags, return `None`
        Given a list `[x]`, simply return `x`
        Given a list `[x, y]`, return "x and y"
        Given a list `[x, y, z]`, convert it to a string `"x, y, and z"
        '''
        # TODO(TG): Again, I have a feeling there's a better way to do this.
        if len(tags) == 0:
            return
        elif len(tags) == 1:
            return tags[0]
        elif len(tags) == 2:
            return f"{tags[0]} and {tags[1]}"
        elif len(tags) < 2:
            return f"{', '.join(tags[:-1])} and {tags[-1]}"
        
    genre_tags = []
    time_tags = []

    for tag in tags:
        if tag in GENRE_TAGS:
            desc = _GENRE_TO_DESC_MAPPING[tag]
            genre_tags += [desc]
        elif tag in TIME_PERIOD_TAGS:
            desc = _TIME_TO_DESC_MAPPING[tag]
            time_tags += [desc]

    return construct_conjunction(genre_tags), construct_conjunction(time_tags)

# Tags.
WRITING_STYLE_TAGS = ["Free", "Casual", "Advanced"]
GENRE_TAGS = ["Horror", "Sci-Fi", "School", "Tabletop", "Nation", "Arena", "Military", "Fantasy", "Romance", "Slice of Life", "Anime/Manga", "Fandom", "Steampunk", "Superhero"]
TIME_PERIOD_TAGS = ["Western", "Ancient", "Apocalyptic", "Post-Apocalyptic", "Historical", "Medieval", "Modern", "Future"]

SYSTEM_PROMPTS = generate_prompts([
    "%{Enter|Engage|Enable|Start} %{fiction writing|fiction|roleplay|RP} mode.",
    "You are now in %{fiction writing|fantasy writing|fiction|roleplay|RP} mode. Drive the story forward in chunks.",
    "You are an %{AI|artificial intelligence} trained to perform %{storywriting|fiction writing|fantasy writing|fantasy roleplay|fiction roleplay|RP}. Generate continuations for whatever the user gives.",
    # Modified SillyTavern prompt
    "Write the next reply in a fictional %{roleplay|RP} %{chat|conversation}.",
    "I am %{in|currently in|engaging in|beginning} a %{roleplay|RP|fictional roleplay-like conversation} with %{someone else|other people|a user}.",
])

# Writing style prompts
FREE_PROMPTS = generate_prompts([
    " %{Write|Compose} in a %{short|brief} and informal %{manner|way}.",
    " Be %{freehand|laid back|informal|casual|relaxed} in terms of %{writing|composition}; don't put too %{much effort|many words} into it.",
    " %{Treat|Take} this as a %{casual|quick|relaxed} %{RP|roleplay} session.",
])

CASUAL_PROMPTS = generate_prompts([
    " Written %{responses|replies} should be of %{medium|moderate|decent} length.",
    " %{Treat|Take} this %{roleplay|RP} somewhat seriously.",
    " %{Responses|Replies} should be at least a few paragraphs in length."
])

ADVANCED_PROMPTS = generate_prompts([
    " %{Write|compose} with heavy detail and make every reply have a long length.",
    " %{Responses|Replies} should be very %{detailed|complex} and contain multiple paragraphs.",
    " %{Treat|Take} this %{roleplay|RP} very seriously; put a lot of effort into %{replies|responses} and make them very long and intricate."
])

STYLE_PROMPT_MAPPING = {
    "Free": FREE_PROMPTS,
    "Casual": CASUAL_PROMPTS,
    "Advanced": ADVANCED_PROMPTS
}

# It's incomplete because the script will finish the rest depending on the time period.
TIME_PROMPTS = generate_prompts([
    " %{The|This} %{roleplay|RP} is set in ",
    " The time period of this %{roleplay|setting|RP} is ",
    " Time period: "
])

GENRE_PROMPTS = generate_prompts([
    " Genre: ",
    " The %{type|genre} of this %{roleplay|RP} is ",
    " The %{themes|genres} are "
])

NSFW_PROMPTS = generate_prompts([
    " %{Generations|Your writing|The generated response|Your reply|Generated replies} must %{be not safe for work|be NSFW|include adult themes|include erotic themes|include 18+ content}",
])

# Genre keyword prompts
_GENRE_TO_DESC_MAPPING = {
    "Horror": "horror",
    "Sci-Fi": "sci-fi",
    "School": "school life",
    "Tabletop": "tabletop games",
    "Nation": "nation-states",
    "Arena": "fighting",
    "Military": "war and the military",
    "Fantasy": "fantasy",
    "Romance": "romance",
    "Slice of Life": "slice of life",
    "Anime/Manga": "anime/manga",
    "Fandom": "an existing fandom",
    "Steampunk": "steampunk",
    "Superhero": "superheroes"
}

_TIME_TO_DESC_MAPPING = {
    "Western": "the time period of the Wild West",
    "Ancient": "ancient times",
    "Apocalyptic": "the apocalypse",
    "Post-Apocalyptic": "after an apocalypse",
    "Historical": "the past",
    "Medieval": "medieval times",
    "Modern": "modern times",
    "Future": "the future"
}

# At least one thread I saw has either been edited post-scrape or something,
# because the entries just say "cut" and are as a result garbage training data.
# Have a variable to sift out threads which consist of only this nonsense.
BROKEN_THREADS = [
    "SAO: Aincrad (1x1 between"
]
