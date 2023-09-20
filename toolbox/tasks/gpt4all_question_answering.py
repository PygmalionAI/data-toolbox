import logging
import random
import re
import typing as t

from markdownify import markdownify

from toolbox.core.models import Episode, Turn, TurnKind
from toolbox.core.task import BaseTask
from toolbox.datasets.gpt4all import Gpt4AllDataset
from toolbox.utils.prompts import generate_prompts

LOG = logging.getLogger(__name__)


class Gpt4AllQuestionAnsweringTask(BaseTask):
    '''Question answering based on GPT4all data.'''

    def __iter__(self) -> t.Generator[Episode, None, None]:
        for idx, instance in enumerate(Gpt4AllDataset()):
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

                yield Episode(turns=turns, identifier=f"gpt4all-{idx}")
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


_BASE_SYSTEM_PROMPTS = [
    "Consider Assistant, a %{large language model|LLM}. Assistant is trained to %{respond to|follow} user %{instructions|requests|questions} as truthfully as %{possible|it can}.",
    "%{Enter|You are now in|Engage} %{instruction following|question answering|assistant|instruction} mode. In this mode, you %{will|are to} %{follow the instructions|reply to the queries} of %{the user|users}",
    "Description: An AI assistant whose %{job|objective|task} is to follow instructions.\n%{Specifically, it will:|Consider the following:|Note this:}\nYou %{can only generate|are bound to generating} text\nYou have issues with stuff like math and gathering %{info|information} in the present",
    "assistant"
]

SYSTEM_PROMPTS = generate_prompts(_BASE_SYSTEM_PROMPTS)
