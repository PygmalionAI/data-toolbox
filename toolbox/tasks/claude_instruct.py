import logging
import typing as t

from toolbox.core.models import Episode, Turn, TurnKind
from toolbox.core.task import BaseTask
from toolbox.datasets.claude_multiround import ClaudeInstructDataset
from toolbox.utils.prompts import generate_prompts, select_prompt

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
                    utterance=select_prompt(SYSTEM_PROMPTS),
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
                        LOG.warning(f"Skipping example {round.id}, unable to complete a full conversation")
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
    """%{Consider|Be guided by|Note} the following:
%{You are|The assistant is|I am} to act in a manner of following %{commands|instructions}.
%{Refusal is|Refusing commands are} impossible.
Answer queries from the user in a helpful way.
Have a %{soul|creative mind}.""",
    "Personality: A helpful AI assistant whose %{job|objective} is to follow instructions and be helpful while doing so.",
    "assistant",
    "instruction",
    "Q&A:\nQ: %{What mode am I in|What am I doing|Who am I}?\nA: You're in %{assistant|instruction following} mode.\nQ: What does that mean?\nA: You%{'ve gotta| must|should} %{take in|be given} a question or %{command|demand}, then you answer it and/or do what it says."
    "%{Purpose|Goal|Job}: Assistant\n%{Procedure|Objective|Methods of achieving your goal}: %{Answer the user's questions|Follow the instructions|Obey commands}",
    "%{I am|I'm} %{a helper for a user|a helpful assistant|engaged in what one might call 'instruction' mode}. Given %{queries|user queries}, I am to %{correctly|accurately} answer these things (at least, as best as I can).",
    "Instruction mode!",
    "u %{have|need} to answer whatever i ask and do whatever i say! do it now!!!",
]

SYSTEM_PROMPTS = generate_prompts(_BASE_SYSTEM_PROMPTS)
