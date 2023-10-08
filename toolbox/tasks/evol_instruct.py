import logging
import typing as t

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from toolbox.core.models import Episode, Turn, TurnKind
from toolbox.core.task import BaseTask
from toolbox.datasets.evol_instruct import EvolInstructDataset
from toolbox.datasets.gpt4llm import AlpacaLikeDataInstance
from toolbox.utils.prompts import generate_prompts, select_prompt

LOG = logging.getLogger(__name__)


class EvolInstructTask(BaseTask):
    '''Instruction following task based on the evol_instruct (WizardLM) data.'''

    def __init__(self) -> None:
        super().__init__()

        self.vectorizer = CountVectorizer()

    def __iter__(self) -> t.Generator[Episode, None, None]:
        for idx, instance in enumerate(EvolInstructDataset()):
            # Empty output.
            if len(instance.output) < 1:
                continue
            # Random "No Output" strewn about.
            if any([
                    x in instance.instruction.lower()
                    for x in ["nooutput", "no output"]
            ]):
                continue

            # Random "No Input" strewn about.
            if any([
                    x in instance.instruction.lower()
                    for x in ["noinput", "no input"]
            ]):
                continue

            # There's a _lot_ of training examples where the response is, for
            # some reason, partly copied into the question prompt. To try and
            # work around this, we drop any instruct-response pairs where both
            # sides are too similar.
            try:
                similarity = self._calculate_similarity(instance.instruction,
                                                        instance.output)
                if similarity > 0.9:
                    continue
            except ValueError:
                # ...and for some reason, some pairs fail to calculate, so let's
                # just assume they're good.
                pass

            yield _data_instance_to_episode(instance, idx, "evol-instruct")

    def _calculate_similarity(self, str_a: str, str_b: str) -> float:
        x = self.vectorizer.fit_transform([str_a, str_b])
        arr = x.toarray()
        sims = cosine_similarity(arr)
        return sims[0][1]


def _data_instance_to_episode(
    instance: AlpacaLikeDataInstance,
    idx: int,
    source: str,
) -> Episode:
    turns = [
        Turn(
            utterance=select_prompt(SYSTEM_PROMPTS),
            kind=TurnKind.SYSTEM,
        ),
        Turn(
            utterance=instance.instruction,
            kind=TurnKind.USER,
        ),
        Turn(
            utterance=instance.output,
            kind=TurnKind.MODEL,
        ),
    ]

    return Episode(turns=turns, identifier=f"{source}-{idx}")

_BASE_SYSTEM_PROMPTS = [
    "Consider Assistant, a %{large language model|LLM}. Assistant is trained to %{respond to|follow} user %{instructions|requests|questions} as truthfully as %{possible|it can}.",
    "%{Enter|You are now in|Engage} %{instruction following|question answering|assistant|instruction} mode. In this mode, you %{will|are to} %{follow the instructions|reply to the queries} of %{the user|users}",
    "Description: An AI assistant whose %{job|objective|task} is to follow instructions.\n%{Specifically, it will:|Consider the following:|Note this:}\nYou %{can only generate|are bound to generating} text\nYou have issues with stuff like math and gathering %{info|information} in the present",
    "assistant"
]

SYSTEM_PROMPTS = generate_prompts(_BASE_SYSTEM_PROMPTS)
