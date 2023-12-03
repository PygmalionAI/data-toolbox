import logging
import re

from typing import Generator, Optional

from markdownify import markdownify

from ..core import (
    BaseFilter,
    BaseTask,
    Episode,
    Turn,
    TurnKind
)
from ..datasets import RpForumsDataset, RpType
from ..utils import (
    fix_style_and_encoding_issues,
    remove_excessive_newlines,
    remove_links,
    remove_mentions,
    remove_ooc,
    remove_trailing_whitespace_and_bad_lines,
    PromptManager
)

LOG = logging.getLogger(__name__)

MARKDOWN_NOSPACE_PATTERN = re.compile(r"([\w\d])(\*{1,2})([\w\d])")
ONLY_OOC_PATTERN = re.compile(r"^\([^)]*\)\.?$")

class RpForumsRoleplayTask(BaseTask):
    '''
    Task to continue a roleplay 
    '''
    def __init__(
        self, 
        filters: list[BaseFilter],
        custom_prompts: Optional[list[str]] = None,
        remove_ooc: bool = False,
        **kwargs
    ) -> None:
        super().__init__(filters=filters)
        self.remove_ooc = remove_ooc
        if custom_prompts is None:
            kwargs = {"custom_prompts": SYSTEM_PROMPTS} if custom_prompts is None \
            else {"custom_prompts": custom_prompts}
        self.prompts = PromptManager(**kwargs)

    def _clean_message(
        self,
        message: str,
        username_subs: dict[str, str]
    ) -> str:
        '''
        Cleans a single message. Best to keep this in its own separate
        function for readability and also so that we can keep it isolated
        from the yielding process, due to having to keep buffers in mind.
        '''
        message = fix_style_and_encoding_issues(message)
        message = _remove_bad_html_tags(message)
        message = remove_links(message)

        # Convert to markdown.
        message = str(markdownify(message))
        message = remove_trailing_whitespace_and_bad_lines(message)
        message = _fix_markdown(message)

        # Excessive newlines
        message = remove_excessive_newlines(message)

        # Username substitutions need to be done _after_ the HTML has
        # been converted into markdown, otherwise we get escape
        # characters messing things up.
        for name, substitution in username_subs.items():
            message = re.sub(rf"\b{re.escape(name)}\b",
                                    substitution, message)
            
        # Remove mentions and OOC if user wants OOC to be purged.
        message = remove_mentions(message)
        if self.remove_ooc:
            message = remove_ooc(message)
            
        return message
    
    def _reset_buffer(self) -> None:
        '''Resets the buffer.'''
        self.previous_author = None
        self.full_post = None
        self.current_kind = TurnKind.USER

    def __iter__(self) -> Generator[Episode, None, None]:
        for thread in RpForumsDataset():
            # Set up a buffer for keeping track of authors.
            # If two posts are made by the author in a row, chain it. This way
            # we avoid repeated human/model turns, which can confuse the model
            # big time.
            self._reset_buffer()

            # These threads usually don't contain actual roleplaying.
            if any([
                    x in thread.thread_name.lower() for x in [
                        "ooc", "o.o.c", "character sheet", "character profile",
                        "character list", "character roster"
                    ]
            ]):
                LOG.debug(f"Skipping {thread.thread_name} due to thread name",
                          )
                continue

            if len(thread.messages) < 2:
                LOG.debug(f'Skipping {thread.thread_name} with only one message',
                          )
                continue

            # If the thread only has one author, no way to tell between human
            # and model turns.
            usernames = set([t.author for t in thread.messages])
            if len(usernames) != 2:
                LOG.debug(f"Skipping {thread.thread_name} that doesn't have 2 authors")
                continue

            # Build up a dictionary of usernames to replace for privacy reasons.
            username_substitutions: dict[str, str] = {}
            for idx, name in enumerate(usernames):
                username_substitutions[name] = "{{char_" + str(idx) + "}}"

            # System prompt
            system_prompt = self.prompts.sample_prompt()
            content_type_prompt = PromptManager(
                CONTENT_TYPE_TO_PROMPTS[thread.content_type]
            ).sample_prompt()
            system_prompt = system_prompt.replace("{{content_type_str}}", \
                content_type_prompt)
            
            # Add system prompt to turn
            turns: list[Turn] = [
                Turn(
                    utterance=system_prompt, kind=TurnKind.SYSTEM
                )
            ]

            # Assign usernames to either the user or the model.
            # Since we've checked beforehand that there's only two authors,
            # we can just assign the first one to the user and the second one
            # to the model.
            user_author = self.previous_author = thread.messages[0].author
            model_author = (usernames - {user_author}).pop()
            roles = {
                user_author: TurnKind.USER,
                model_author: TurnKind.MODEL
            }

            for message in thread.messages:
                # Check the author, if it's *not* the same as the previous
                # author then yield the full turn. This should be done *first*.
                if message.author != self.previous_author and \
                self.full_post is not None:
                    turns.append(
                        Turn(
                            utterance=self.full_post,
                            kind=roles[self.previous_author],
                            # TODO(TG): Assign a proper name.
                            name="TODO"
                        )
                    )
                    self.previous_author = message.author
                    self.full_post = ""

                # Now that we got past the first check, empty the string.
                if self.full_post is None:
                    self.full_post = ""

                # Process the message.
                cleaned_message = self._clean_message(
                    message.message, 
                    username_substitutions
                )

                self.full_post = (self.full_post + f"\n{cleaned_message}").strip()

            # Yield the final turn.
            final_kind = roles[thread.messages[-1].author]
            turns.append(
                Turn(
                    utterance=self.full_post,
                    kind=final_kind,
                    name="TODO"
                )
            )

            # Sometimes we just have a situation where the HTML cleaning
            # results in a faulty message.
            # If this is the case for every message, just ditch the thread.
            if _thread_unsalvagable(turns[1:]):
                LOG.info(f"Skipping {thread.thread_name} due to being deemed 'unsalvagable'")
                continue

            # Update the system prompt by filling in the template strings.
            turns[0].utterance = PromptManager.fill_response_style_length(
                turns[0].utterance, self.full_post)

            yield Episode(
                turns=turns,
                identifier=f"rp-{thread.source_file}-{thread.thread_name}"
            )

def _failed_cleaning(message: str) -> bool:
    '''
    Sometimes markdownify, HTML tag removal and additional processing results
    in a message which has nothing left, likely due to faulty formatting.
    This function attempts to detect those situations, or other situations
    '''
    if len(message.strip()) <= 1:
        return True
    # OOC only.
    if ONLY_OOC_PATTERN.search(message) is not None:
        return True
    return False

def _thread_unsalvagable(turns: list[Turn], threshold: float = 0.5) -> bool:
    '''
    If the thread is messy enough that we can't salvage a threshold of messages,
    then we just ditch the thread. By default, it's 50%.
    '''
    # Fun fact: True == 1 in Python, so we can just sum up the booleans.
    return sum(_failed_cleaning(x.utterance) for x in turns) / len(turns) >= threshold

# Unique functions (for now...)
def _fix_markdown(original_message: str) -> str:
    s = original_message

    # Bold/italics sometimes doesn't have spaces around it after converting from
    # HTML to Markdown for some reason.
    is_opening_asterisk = True
    while (match := MARKDOWN_NOSPACE_PATTERN.search(s)) is not None:
        if is_opening_asterisk:
            s = s[:match.start() + 1] + " " + s[match.start() + 1:]
        else:
            s = s[:match.end() - 1] + " " + s[match.end() - 1:]
        is_opening_asterisk = not is_opening_asterisk

    return s


def _remove_bad_html_tags(message: str) -> str:
    '''Cleans up HTML tags we don't want from the given message.'''
    cleaned_message = _remove_html_tag(message, "blockquote")
    cleaned_message = _remove_html_tag(cleaned_message, "script")

    if "bbImageWrapper" in message:
        # Images are a <div> with some JavaScript to lazy-load them, so we do
        # this behind a guard to reduce false positives just in case.
        cleaned_message = _remove_html_tag(cleaned_message, "div")

    return cleaned_message


def _remove_html_tag(message: str, tag: str) -> str:
    '''Cleans the given HTML tag from the message.'''
    cleaned_message = message
    cleaning_passes = 0

    while f"<{tag}" in cleaned_message:
        assert cleaning_passes < 4, "Too many cleaning passes, giving up to avoid deadlocking"

        start_idx = cleaned_message.find(f"<{tag}")
        end_idx = cleaned_message.find(f"</{tag}>", start_idx)

        if start_idx == -1 or end_idx == -1:
            LOG.warning("Unbalanced tags found, leaving as-is")
            break

        cleaned_message = cleaned_message[:start_idx] + cleaned_message[
            end_idx + len(f"</{tag}>"):]

    return cleaned_message

# Constants

SYSTEM_PROMPTS = [
    '''%{Enter|Engage|Enable|Start} %{fiction writing|fantasy writing|fantasy roleplay|fictional RP|roleplay|RP} mode. {{content_type_str}}. {{response_length_str}}.''',
    #
    '''You %{are now in|have entered|will now start} %{fiction writing|fantasy writing|fantasy roleplay|fictional RP|roleplay|RP|conversational RP} mode. Drive the story forward in chunks. {{response_length_str}}.''',
    #
    '''You are trained to %{perform|generate} %{storywriting|fiction writing|fantasy writing|fantasy roleplay|fictional roleplay|RP}. Generate continuations for whatever the user gives. {{response_length_str}}. {{content_type_str}}.''',
    # Modified SillyTavern prompt
    '''Write the next reply in a fictional %{roleplay|RP} %{chat|conversation}. {{content_type_str}}. {{response_length_str}}.''',
    #
    '''%{SYSTEM|MODE}: %{conversational roleplay|RP|roleplay mode|RP system engaged}
%{NOTE|ADVISORY|KEEP IN MIND}: {{response_length_str}}''',
    #
    '''I am %{in|currently in|engaging in|beginning} a %{roleplay|RP|fictional roleplay-like conversation} with %{someone else|other people|a user}.''',
    #
    '''{{content_type_str}}. {{response_length_str}}.''',
    #
    '''%{OBJECTIVE|TASK|MISSION|JOB} - %{Conduct|Generate|Enjoy} a %{roleplay session|RP|fictional roleplay}
%{DISCRETION RATING|SAFE FOR WORK?|CONTENT RATING} - {{content_type_str}}
%{REMEMBER|NOTE} - {{response_length_str}}''',
    # Misspellings intentional
    '''%{do|make|have} %{rp adventures|writing|creative roleplay}
%{pls|please} %{rember|remember} to %{b|be} %{engaging|immersive|epic}''',
    #
]

SFW_PROMPTS = [
    "%{Generations|Your writing|The generated response|Your reply|Generated replies} must %{be safe for work|be SFW|not include any adult themes|be safe for minors|not include 18+ content|not be 18+|not be NSFW}",
]

MIXED_SFW_NSFW_PROMPTS = [
    "%{Generations|Your writing|The generated response|Your reply|Generated replies} %{may or may not include adult themes|may or may not be NSFW|can include adult themes}",
]

NSFW_PROMPTS = [
    "%{Generations|Your writing|The generated response|Your reply|Generated replies} must %{be not safe for work|be NSFW|include adult themes|include erotic themes|include 18+ content}",
]

CONTENT_TYPE_TO_PROMPTS: dict[RpType, list[str]] = {
    RpType.RP: SFW_PROMPTS,
    RpType.ERP: NSFW_PROMPTS,
    RpType.MIXED: MIXED_SFW_NSFW_PROMPTS,
}