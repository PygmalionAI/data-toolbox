import logging
import re
import typing as t

from toolbox.core.models import Episode, Turn, TurnKind
from toolbox.core.task import BaseTask
from toolbox.datasets.whocars import WhocarsDataset

LOG = logging.getLogger(__name__)

# A minor note for this task: the data does not seem to be very clean. Even
# GPT-4 seems to have trouble following the system prompt, resulting in
# instructions like "ALWAYS precede dialogue with character names" being
# ignored. Pronouns are also messed up sometimes. This will likely bleed into
# our model, but for now I'm not gonna bother with this.


class WhocarsRoleplayTask(BaseTask):
    '''Task to roleplay as a given character.'''

    def __iter__(self) -> t.Generator[Episode, None, None]:
        for idx, entry in enumerate(WhocarsDataset()):
            if entry.endpoint == "kobold":
                continue

            assert entry.endpoint == "openai", entry.endpoint
            if "gpt-4" not in entry.model:
                continue

            if entry.prompt_json[0]["role"] != "system":
                continue

            turns: list[Turn] = []
            for msg in entry.prompt_json:
                utterance = msg["content"].strip()

                turn_kind = TurnKind.MODEL
                if msg["role"] == "system":
                    turn_kind = TurnKind.SYSTEM
                    utterance = _clean_system_message(utterance)
                if msg["role"] == "user":
                    turn_kind = TurnKind.USER

                turn = Turn(
                    utterance=_clean_message(utterance),
                    kind=turn_kind,
                )
                turns.append(turn)
            yield Episode(turns=turns, identifier=f"whocars-{idx}")


def _clean_system_message(msg: str) -> str:
    # TavernAI's system message(s) very often refer to the user as You, but
    # uses a dumb string replace which means there's broken grammar and
    # conflicting instructions within the prompt usually. To try and alleviate
    # that, we replace `You` with `{{user}}` for clarity.
    return re.sub(r"\bYou\b", "{{user}}", msg)


def _clean_message(msg: str) -> str:
    '''Handles common typos or bad tags.'''
    msg = msg.replace("{{chaar}}", "{{char}}")
    msg = msg.replace("{{character}}", "{{char}}")
    return msg
