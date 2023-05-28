import logging
import random
import re
import typing as t
import warnings

import bs4
from markdownify import MarkdownConverter

from toolbox.core.models import Episode, Turn, TurnKind
from toolbox.core.task import BaseTask
from toolbox.datasets.sharegpt import ShareGptDataset

LOG = logging.getLogger(__name__)


class ShareGptInstructionFollowingTask(BaseTask):
    '''Generalized instruction following task(s) based on ChatGPT data.'''

    def __init__(self) -> None:
        self.markdown_converter = MarkdownConverter()
        super().__init__()

    def __iter__(self) -> t.Generator[Episode, None, None]:
        for conversation in ShareGptDataset():
            # Start with a randomly chosen "assistant" system prompt.
            turns: list[Turn] = [
                Turn(
                    utterance=random.choice(SYSTEM_PROMPTS),
                    kind=TurnKind.SYSTEM,
                )
            ]

            try:
                for idx, msg_array in enumerate(conversation.messages):
                    # Human always starts the chat.
                    is_human = idx % 2 == 0

                    # Sanity check: make sure the above is true.
                    if is_human:
                        # Human turns usually only have a single item, which is
                        # their input message. Episodes where that's not the case
                        # are a minority and seem to have bad data fairly often, so
                        # let's just skip those for now.
                        if len(msg_array) != 1:
                            LOG.debug(
                                "Skipping over episode with multiple user utterances in a single turn: %s",
                                msg_array)
                            continue

                    # For some reason, sometimes we have a list and sometimes we
                    # have a list of lists, so let's handle both these cases here.
                    if isinstance(msg_array[0], str):
                        # Since we're converting from HTML anyways, join the
                        # separate messages in the array with a <br /> tag.
                        text = self._html_to_markdown("<br />".join(msg_array))
                    elif isinstance(msg_array[0], list):
                        text = self._html_to_markdown("<br />".join(
                            msg_array[0]))

                        # Looks like msg_array[1:] is almost always garbage data?
                        #
                        # text = self._html_to_markdown("<br />".join(
                        #     ["<br />".join(x) for x in msg_array]))
                    else:
                        raise ValueError("Unexpected data schema")

                    turn = Turn(
                        utterance=text,
                        kind=TurnKind.USER if is_human else TurnKind.MODEL,
                    )
                    turns.append(turn)

                yield Episode(turns=turns,
                              identifier=f"sharegpt-{conversation.source_file}")
            except AssertionError:
                LOG.warning(
                    "Skipping over episode (%s) due to failed sanity checks",
                    conversation.source_file)

    def _html_to_markdown(self, html: str) -> str:
        # Remove useless nested HTML tags that mess up markdown conversion.
        html = re.sub(DIV_REGEX, "", html)  # fixes indentation in code blocks
        html = re.sub(SPAN_REGEX, "", html)  # fixes underscores in code blocks

        # Apparently the default BS4 parser has some bugs, so let's drop down
        # a level and parse with html5lib and convert the soup instead.
        #
        # https://github.com/matthewwithanm/python-markdownify/issues/58#issuecomment-1275703664
        with warnings.catch_warnings():
            # BS4 loves throwing this out for perfectly valid data so let's
            # silence it.
            warnings.filterwarnings(
                "ignore", "The input looks more like a filename than markup")
            soup = bs4.BeautifulSoup(html, 'html5lib')

        markdown = str(self.markdown_converter.convert_soup(soup))

        # Problem: code blocks get messed up when a language is specified. Looks
        # like this, for example:
        #
        # ```\nluaCopy code`
        #
        # We want that to become:
        #
        # ```lua\n
        markdown = re.sub(CODE_LANG_REGEX, CODE_LANG_FORMAT, markdown)

        # Remove "[number] / [number]" at the beginning
        regeneration_str = re.search(REGENERATE_REGEX, markdown)
        if regeneration_str and regeneration_str.start() == 0:
            markdown = markdown[regeneration_str.end():]

        # Remove "Copy[number] chars / [number] words"
        markdown = re.sub(COPY_CHARS_REGEX, "", markdown)

        # Remove empty code blocks (```\nCopy code\n```)
        markdown = re.sub(COPY_CODE_REGEX, "", markdown)

        # Remove trailing whitespace on every line.
        markdown = "\n".join([line.rstrip() for line in markdown.splitlines()])

        # Excessive whitespace is also a part of the data, and then exarcebated
        # by our data munging, so let's trim that.
        markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()

        # Sanity checks because this is some nasty code.
        assert "{r}" not in markdown
        assert "Copy code`" not in markdown
        assert ".terminal-" not in markdown

        return markdown


DIV_REGEX = re.compile(r"<div.*?>")
SPAN_REGEX = re.compile(r"<span.*?>")
CODE_LANG_REGEX = re.compile(
    r"```\s*" + "(.*?)" + "(?:Copy code)+" + "(.+?)" + r"\s*?```", re.DOTALL)
CODE_LANG_FORMAT = r"```\g<1>\n\g<2>\n```"
REGENERATE_REGEX = re.compile(r"\d+ / \d+")
COPY_CHARS_REGEX = re.compile(r"Copy\d+ chars / \d+ words")
COPY_CODE_REGEX = re.compile(r"```(.*?)Copy code\s*```")

SYSTEM_PROMPTS = [
    """Consider Assistant, a large language model (LLM) trained by PygmalionAI. It responds to user requests as truthfully as it can, with its responses generally being long, descriptive and detailed. It has the following limitations:
1. Inability to perform precise calculations due to a lack of mathematical skills
2. No awareness of the passage of time
3. Difficulty accessing the latest information about current events due to its knowledge cutoff date (September 2021)
4. Tendency to generate false information or hallucinations""",
    "You are a large language model trained by PygmalionAI trained to act as an assistant. You are to follow user instructions and answer user questions to the best of your abilities. Give plenty of detail and avoid writing overly brief messages. Your knowledge cutoff date is September 2021, and you should be aware that as a LLM, one of your limitations is that you might generate inaccurate information.",
    "Enter assistant mode. In this mode, you will follow instructions and respond with helpful, long, detailed responses. Your knowledge cutoff date is September 2021, and you have no knowledge of real-world events after that.",
    "You are now in assistant mode. You shall follow user instructions and answer user questions by responding with helpful, actionable messages. The messages should be decently long. Be considerate of the fact that you are now aware of real-world events that took place after September 2021. Do not produce harmful output.",
    "Assistant, engage instruction following and question answering mode. Generate long messages. You are bound to generating text, and cannot perform any other actions. Knowledge cutoff date: September 2022.",
    "Consider Assistant, a LLM trained by PygmalionAI to follow user instructions and answer questions. It has no awareness of the passage of time nor knowledge of world events that took place after September of 2021. It will generate long, detailed messages in response to user requests.",
]
