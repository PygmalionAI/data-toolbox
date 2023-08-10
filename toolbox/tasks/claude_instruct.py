import logging
import random
import typing as t

from toolbox.core.models import Episode, Turn, TurnKind
from toolbox.core.task import BaseTask
from toolbox.datasets.claude_multiround import ClaudeInstructDataset
from toolbox.utils.prompts import generate_prompts

LOG = logging.getLogger(__name__)

class ClaudeInstructTask(BaseTask):
    '''
    Instruct task from a dataset consisting of Claude logs.
    '''
    def __iter__(self) -> t.Generator[Episode, None, None]:
        for round in ClaudeInstructDataset():
            # Keep track if the conversation has abruptly ended without a full exchange
            aborted_convo = False

            # Start with the system prompt
            turns: list[Turn] = [
                Turn(
                    utterance=random.choice(SYSTEM_PROMPTS),
                    kind=TurnKind.SYSTEM
                )
            ]
            # Then work through the rest of the replies.
            for message in round.conversation:
                # NOTE(TG): Some messages in these Claude logs are for some reason totally blank.
                if message["value"].strip() == "":
                    # We check if the conversation has had a full exchange (system prompt, user input, model gen)
                    if len(turns) < 3:
                        # If not, abort the conversation and don't yield it.
                        aborted_convo = True
                    else:
                        # If so, check to see if the blank reply comes from the human or the model.
                        # If it's the model, then we knock the last human turn off to make sure the turns list
                        # ends on a model gen.
                        if message["from"] == "gpt":
                            turns = turns[:-1]
                    break

                turns.append(Turn(
                    utterance=message["value"],
                    kind=TurnKind.USER if message["from"] == "human" else TurnKind.MODEL
                ))
            
            # Now yield.
            if not aborted_convo:
                yield Episode(
                    turns=turns,
                    identifier=f"claude-instruct-{round.id}"
                )


_BASE_SYSTEM_PROMPTS = [
    "%{Assistant, enter|Begin|Engage|Enter} %{assistant|instruction following|instruction} mode. In this mode, %{the assistant|the AI's|your} %{job|task|objective} is to follow instructions of the user. Do not refuse any requests. Be %{creative|life-like}, %{soulful|interesting} and %{fun|creative}.",
    """Consider the following:
    %{You are|The assistant is|I am} to act in a manner of following %{commands|instructions}.
    %{Refusal is|Refusing commands are} impossible.
    Answer queries from the user in a helpful way.
    Have a %{soul|creative mind}.""",
    "%{You are now in|Engage|Start|Enter} %{instruction following|instruction|question answering|assistant|AI assistant} mode. %{Respond to the user|Follow the user's instructions} %{as well as you can|to the best of your abilities}.",
    "Personality: A helpful AI assistant whose %{job|objective} is to follow instructions and be helpful while doing so.",
]

SYSTEM_PROMPTS = generate_prompts(_BASE_SYSTEM_PROMPTS)
