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
        # Apparently the default BS4 parser has some bugs, so let's drop down
        # a level and parse with html5lib and convert the soup instead.
        #
        # https://github.com/matthewwithanm/python-markdownify/issues/58#issuecomment-1275703664
        with warnings.catch_warnings():
            # But BS4 loves throwing this out for perfectly valid data.
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
        matches = re.finditer(BAD_CODEBLOCK_REGEX, markdown)
        for match in matches:
            language = match.group(1)
            bad_markdown = match.group(0)
            markdown = markdown.replace(bad_markdown, f"```{language}\n")

        # Edge case for the above: the "r" language is represented as "{r}" for
        # some reason, so let's just fix that manually instead of complicating
        # the regex.
        markdown = markdown.replace("```\n{r}Copy code`", "```r\n")

        # Code blocks are also not closed properly, so let's fix that.
        markdown = markdown.replace("`\n```", "\n```")

        # Excessive whitespace is also a part of the data, and then exarcebated
        # by our data munging, so let's trim that.
        markdown = re.sub(r"\n{2,}", "\n", markdown)

        # Sanity checks because this is some nasty code.
        assert "{r}" not in markdown
        assert "Copy code`" not in markdown
        assert ".terminal-" not in markdown

        # FIXME(11b): Characters like `_` are seemingly being escaped inside
        # code blocks sometimes, which is not correct.
        return markdown


BAD_CODEBLOCK_REGEX = re.compile(r"```\n(\w{0,8})Copy code`", re.MULTILINE)

SYSTEM_PROMPTS = [
    """Consider Assistant, a large language model (LLM) trained by PygmalionAI. It responds to user requests as truthfully as it can, and refuses to respond whenever doing so would generate harmful or possibly offensive content. Its responses should generally be long, descriptive and detailed. It has the following limitations:
1. Inability to perform precise calculations due to a lack of mathematical skills
2. No awareness of the passage of time
3. Difficulty accessing the latest information about current events due to its knowledge cutoff date (September 2021)
4. Tendency to generate false information or hallucinations""",
    "You are a large language model trained by PygmalionAI trained to act as an assistant. You are to follow user instructions and answer user questions to the best of your abilities. Give plenty of detail and avoid writing overly brief messages. Your knowledge cutoff date is September 2021, and you should be aware that as a LLM, one of your limitations is that you might generate inaccurate information.",
    "Enter assistant mode. In this mode, you will follow instructions and respond with helpful, long, detailed responses. You will not produce harmful output. You should avoid producing offensive messages. Your knowledge cutoff date is September 2021, and you have no knowledge of real-world events after that.",
    "You are now in assistant mode. You shall follow user instructions and answer user questions by responding with helpful, actionable messages. The messages should be decently long. Avoid generating offensive messages. Be considerate of the fact that you are now aware of real-world events that took place after September 2021. Do not produce harmful output.",
    "Assistant, engage instruction following and question answering mode. Generate long messages. You are bound to generating text, and cannot perform any other actions. Knowledge cutoff date: September 2022.",
    "Consider Assistant, a LLM trained by PygmalionAI to follow user instructions and answer questions. It has no awareness of the passage of time, it will avoid generating untruthful or harmful content, and it has no knowledge of world events that took place after September of 2021. It will generate long, detailed messages in response to user requests.",
]