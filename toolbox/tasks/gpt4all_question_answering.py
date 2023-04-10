import logging
import random
import re
import typing as t

from markdownify import markdownify

from toolbox.core.models import Episode, Turn, TurnKind
from toolbox.core.task import BaseTask
from toolbox.datasets.gpt4all import Gpt4AllDataset

LOG = logging.getLogger(__name__)


class Gpt4AllQuestionAnsweringTask(BaseTask):
    '''Question answering based on GPT4all data.'''

    def __iter__(self) -> t.Generator[Episode, None, None]:
        for instance in Gpt4AllDataset():
            try:
                turns: list[Turn] = [
                    Turn(
                        utterance=random.choice(SYSTEM_PROMPTS),
                        kind=TurnKind.SYSTEM,
                    ),
                    Turn(
                        utterance=_html_to_clean_markdown(instance.prompt),
                        kind=TurnKind.USER,
                    ),
                    Turn(
                        utterance=_html_to_clean_markdown(instance.response),
                        kind=TurnKind.MODEL,
                    ),
                ]

                yield Episode(turns=turns, identifier="gpt4all")
            except AssertionError as ex:
                # TODO(11b): markdownify lib is choking when seeing some
                # regexes in the data. Skiping data for now, but ideally we'd
                # work around this.
                LOG.warning(
                    "Skipping over data instance due to failed assertion: %s",
                    ex)


def _html_to_clean_markdown(html: str) -> str:
    '''
    Converts the given HTML to Markdown and cleans up any weird-looking stuff
    left behind. Manually identified by randomly sampling the data.
    '''
    markdown = markdownify(html)

    # Fix excessive spaces after converting to Markdown.
    markdown = re.sub("\n{2,}", "\n\n", markdown)

    return markdown.strip()


SYSTEM_PROMPTS = [
    "Consider Assistant, a large language model (LLM) trained by PygmalionAI. It responds to user requests as truthfully as it can, and refuses to respond whenever doing so would generate harmful or possibly offensive content. Its responses should generally be long, descriptive and detailed.",
    "You are a large language model trained by PygmalionAI trained to act as an assistant. You are to follow user instructions and answer user questions to the best of your abilities. Give plenty of detail and avoid writing overly brief messages.",
    "Enter assistant mode. In this mode, you will follow instructions and respond with helpful, long, detailed responses. You will not produce harmful output. You should avoid producing offensive messages.",
    "You are now in assistant mode. You shall follow user instructions and answer user questions by responding with helpful, actionable messages. The messages should be decently long. Avoid generating offensive messages.",
    "Assistant, engage instruction following and question answering mode. Generate long messages. You are bound to generating text, and cannot perform any other actions.",
    "Consider Assistant, a LLM trained by PygmalionAI to follow user instructions and answer questions. It has no awareness of the passage of time, it will avoid generating untruthful or harmful content. It will generate long, detailed messages in response to user requests.",
]