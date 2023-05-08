import logging
import random
import typing as t

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from toolbox.core.models import Episode, Turn, TurnKind
from toolbox.core.task import BaseTask
from toolbox.datasets.evol_instruct import EvolInstructDataset
from toolbox.datasets.gpt4llm import AlpacaLikeDataInstance

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
            utterance=random.choice(SYSTEM_PROMPTS),
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


SYSTEM_PROMPTS = [
    "Consider Assistant, a large language model (LLM). It responds to user requests as truthfully as it can.",
    "You are a large language model trained to act as an assistant. You are to follow user instructions and answer user questions to the best of your abilities.",
    "Enter assistant mode. In this mode, you will follow instructions and respond with helpful responses.",
    "You are now in assistant mode. You shall follow user instructions and answer user questions by responding with helpful, actionable messages.",
    "Assistant, engage instruction following and question answering mode. You are bound to generating text, and cannot perform any other actions.",
    "Consider Assistant, a LLM trained to follow user instructions and answer questions.",
]