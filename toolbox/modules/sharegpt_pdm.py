import random
import typing as t
import re
import logging

import bs4
from markdownify import MarkdownConverter

from toolbox.core.consts import PromptConstants
from toolbox.core.models import Episode, Turn
from toolbox.datasets.sharegpt import ShareGPTDataset
from toolbox.modules import BaseModule

LOG = logging.getLogger(__name__)


class ShareGPTPDM(BaseModule):
    '''Persona Dialogue Module based on ChatGPT data.'''

    SYNTHETIC_PERSONAS = [
        "An artificial intelligence that can generate meaningful responses in plain English",
        "A helpful AI assistant", "Very intelligent", "Extremely knowledgeable",
        "A large language model (LLM) capable of answering user questions",
        "Highly knowledgeable and intelligent AI assistant",
        "Does their best to answer the user's questions",
        "Knows a lot, and always attempts to tell the truth",
        "Gives lengthy, extremely detailed responses",
        "In-depth, detailed answers"
    ]

    def __init__(self) -> None:
        self.markdown_converter = MarkdownConverter()
        super().__init__()

    def generator(self) -> t.Generator[Episode, None, None]:
        for episode in ShareGPTDataset():
            turns: list[Turn] = []
            persona = {
                PromptConstants.BOT_TOKEN:
                    self._generate_synthetic_persona_string()
            }

            for idx, msg_array in enumerate(episode.messages):
                # Human always starts the chat.
                is_human_speaker = idx % 2 == 0

                # Sanity check: make sure the above is true.
                if is_human_speaker:
                    # Human turns only have a single item, which is their input
                    # message. Episodes where that's not the case seem like
                    # badly parsed data.
                    if len(msg_array) != 1:
                        LOG.warning(
                            "Skipping over episode with multiple user utterances in a single turn: %s",
                            msg_array)
                        continue

                if isinstance(msg_array[0], str):
                    text = self._html_to_markdown("<br />".join(msg_array))
                elif isinstance(msg_array[0], list):
                    text = self._html_to_markdown("<br />".join(msg_array[0]))

                    # Looks like msg_array[1:] is always useless garbage?
                    #
                    # text = self._html_to_markdown("<br />".join(
                    #     ["<br />".join(x) for x in msg_array]))
                else:
                    raise ValueError("Unexpected data schema")

                turn = Turn(utterance=text,
                            speaker=PromptConstants.USER_PREFIX
                            if is_human_speaker else PromptConstants.BOT_TOKEN,
                            human_speaker=is_human_speaker)
                turns.append(turn)

            yield Episode(turns=turns, participant_personas=persona)

    def _html_to_markdown(self, html: str) -> str:
        # Apparently the default BS4 parser has some bugs, so let's drop down
        # a level and parse with html5lib and convert the soup instead.
        #
        # https://github.com/matthewwithanm/python-markdownify/issues/58#issuecomment-1275703664
        soup = bs4.BeautifulSoup(html, 'html5lib')
        markdown = self.markdown_converter.convert_soup(soup)

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
        try:
            assert "{r}" not in markdown
            assert "Copy code" not in markdown
        except AssertionError as ex:
            LOG.warning("Assertion failure: %s", ex)

        # FIXME(11b): Characters like `_` are seemingly being escaped inside
        # code blocks sometimes, which is not correct.
        return markdown

    def _generate_synthetic_persona_string(self) -> str:
        selected_personas = random.sample(self.SYNTHETIC_PERSONAS, 5)
        random.shuffle(selected_personas)
        return ". ".join(selected_personas) + "."


BAD_CODEBLOCK_REGEX = re.compile(r"```\n(\w{0,8})Copy code`", re.MULTILINE)
