import logging
import random
import re
import typing as t

from markdownify import markdownify

from toolbox.core.models import Episode, Turn, TurnKind
from toolbox.core.task import BaseTask
from toolbox.datasets.rp_forums import RpForumsDataset, RpType
from toolbox.utils.prompts import generate_prompts, select_prompt

LOG = logging.getLogger(__name__)


class RpForumsWritingTask(BaseTask):
    '''
    Task to generate an appropriate continuation in the context of a fantasy
    roleplay.
    '''

    def __init__(self, all_model_turns: bool = False) -> None:
        # Keep the old way of having the turns be almost entirely
        # model turns, just in case.
        self.all_model_turns = all_model_turns

    def __iter__(self) -> t.Generator[Episode, None, None]:
        for thread in RpForumsDataset():
            # These threads usually don't contain actual roleplaying.
            if any([
                    x in thread.thread_name.lower() for x in [
                        "ooc", "o.o.c", "character sheet", "character profile",
                        "character list", "character roster"
                    ]
            ]):
                LOG.debug("Skipping `%s` due to thread name",
                          thread.thread_name)
                continue

            if len(thread.messages) < 2:
                LOG.debug('Skipping `%s` with only one message',
                          thread.thread_name)
                continue

            # Build up a dictionary of usernames to replace for privacy reasons.
            usernames = set([message.author for message in thread.messages])
            username_substitutions: dict[str, str] = {}
            for idx, name in enumerate(usernames):
                username_substitutions[name] = "{{char_" + str(idx) + "}}"

            # System prompt
            system_prompt = select_prompt(SYSTEM_PROMPTS)
            content_type_prompt = select_prompt(
                CONTENT_TYPE_TO_PROMPTS[thread.content_type])
            system_prompt = system_prompt.replace("{{content_type_str}}",
                                                  content_type_prompt)
            system_turn = Turn(utterance=system_prompt, kind=TurnKind.SYSTEM)
            turns: list[Turn] = [system_turn]

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
                        
                    # NOTE(TG): 11b's original idea where RP generations were framed
                    # as almost entirely model turns in order to get as much data from
                    # it as possible was nice, but a little flawed. In 7B and 13B models,
                    # this caused the model to endlessly ramble on. I'll keep the old code
                    # here, but only if it's manually enabled.
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

                    turn = Turn(utterance=cleaned_message, kind=turn_kind)
                    turns.append(turn)
                
                # Messy switching
                current_speaker = TurnKind.MODEL if current_speaker == TurnKind.USER else TurnKind.USER

            yield Episode(
                turns=turns,
                identifier=f"rp-{thread.source_file}-{thread.thread_name}",
            )


def _split_message(original_message: str, target_word_count: int,
                   delimiter: str) -> list[str]:
    '''
    Splits a large message into smaller ones, respecting the given delimiter.
    '''
    messages = original_message.split(delimiter)
    reconstructed_messages: list[str] = [messages[0]]

    # For each split message, we see if we can merge it back up together with
    # the next one while still staying under the target word count.
    for message in messages[1:]:
        last_message_word_count = len(reconstructed_messages[-1].split()) \
            if len(reconstructed_messages) else 0
        current_message_word_count = len(message.split())

        if last_message_word_count + current_message_word_count > target_word_count:
            # If we can't, we just add it as a separate message to start merging
            # from scratch.
            reconstructed_messages.append(message)
        else:
            # Otherwise, we merge it into the current message.
            reconstructed_messages[-1] += delimiter + message

    return reconstructed_messages


def _fix_style_and_encoding_issues(original_message: str) -> str:
    '''Cleans up any style-related issues.'''
    message = original_message
    message = message.replace(" .. ", "... ")
    message = message.replace(" ... ", "... ")
    message = re.sub(r'\b(\.\.\.?)\b', '... ', message)

    message = message.replace(" . ", ". ")
    message = message.replace(" , ", ", ")
    message = message.replace(" ? ", "? ")
    message = message.replace(" ! ", "! ")

    message = re.sub(r"(\S)(…)(\S)", "\\1\\2 \\3", message)

    # Some forums have their pages incorrectly tagged as UTF-8, so we get
    # garbage when decoding. Most common problem I've seen is bad quotation
    # marks, so we paper over that here.
    message = message.replace("â??", "'")
    message = message.replace("â?", "'")

    message = message.replace("", " ")

    return message


def _remove_links(original_message: str) -> str:
    '''Removes any links from the given message, due to privacy concerns.'''
    return re.sub(r"https?:\/\/.+?(\s|$)", "", original_message)


def _remove_trailing_whitespace_and_bad_lines(original_message: str) -> str:
    lines: list[str] = []
    for line in original_message.splitlines():
        # Trailing whitespace is always useless.
        line = line.rstrip()

        # Sometimes, users start their messages with "RE: (thread title, which
        # leaks usernames)" so we skip that here.
        if line.startswith("RE: ") or line.startswith("**RE: "):
            continue

        lines.append(line)

    return "\n".join(lines)


def _not_usable_as_training_label(message: str) -> bool:
    '''
    Whether or not the message contains some problem that we can't fix reliably,
    and we're better off not training on.
    '''

    # "Floating" quotation marks.
    if re.search(r'\b " \b', message) is not None:
        return True

    # Quotation marks mushed together with text.
    if re.search(r'\S"\S', message) is not None:
        return True

    # Parenthesis mushed together with text.
    if re.search(r'\S\(', message) is not None \
        or re.search(r'\)\S', message) is not None:
        return True

    # Lowercase "I". Fixable, but a sign of low-quality writing so I'd rather
    # not train the model on these.
    if re.search(r"\bi('m|'ll)?\b", message) is not None:
        return True

    # Links.
    if re.search(r"\[.+\]\(\S+\)", message) is not None:
        return True

    return False


def _fix_markdown(original_message: str) -> str:
    s = original_message

    # Bold/italics sometimes doesn't have spaces around it after converting from
    # HTML to Markdown for some reason.
    is_opening_asterisk = True
    while (match := re.search(r"([\w\d])(\*{1,2})([\w\d])", s)) is not None:
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


def _seems_to_have_ooc_talk(message: str) -> bool:
    '''Returns whether a message seems to have some out-of-character talk.'''
    return re.search(_OOC_REGEX, message) is not None


_OOC_REGEX = re.compile(r"^\((OOC: ?)?.+\)$", flags=re.MULTILINE)

_BASE_SYSTEM_PROMPTS = [
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
    '''{{content_style_str}}. {{response_length_str}}.''',
    #
    '''%{OBJECTIVE|TASK|MISSION|JOB} - %{Conduct|Generate|Enjoy} a %{roleplay session|RP|fictional roleplay}
%{DISCRETION RATING|SAFE FOR WORK?|CONTENT RATING} - {{content_type_str}}
%{REMEMBER|NOTE} - {{response_length_str}}''',
    # Misspellings intentional
    '''%{do|make|have} %{rp adventures|writing|creative roleplay}
%{pls|please} %{rember|remember} to %{b|be} %{engaging|immersive|epic}''',
    #
    "%{roleplay|RP}",
    ""
]

SYSTEM_PROMPTS = generate_prompts(_BASE_SYSTEM_PROMPTS)

SFW_PROMPTS = generate_prompts([
    "%{Generations|Your writing|The generated response|Your reply|Generated replies} must %{be safe for work|be SFW|not include any adult themes|be safe for minors|not include 18+ content|not be 18+|not be NSFW}",
])

MIXED_SFW_NSFW_PROMPTS = generate_prompts([
    "%{Generations|Your writing|The generated response|Your reply|Generated replies} %{may or may not include adult themes|may or may not be NSFW|can include adult themes}",
])

NSFW_PROMPTS = generate_prompts([
    "%{Generations|Your writing|The generated response|Your reply|Generated replies} must %{be not safe for work|be NSFW|include adult themes|include erotic themes|include 18+ content}",
])

CONTENT_TYPE_TO_PROMPTS: dict[RpType, list[str]] = {
    RpType.RP: SFW_PROMPTS,
    RpType.ERP: NSFW_PROMPTS,
    RpType.MIXED: MIXED_SFW_NSFW_PROMPTS,
}
