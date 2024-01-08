import logging

from typing import Generator, Optional

from ..core import (
    BaseFilter,
    BaseTask,
    Episode,
    Turn,
    TurnKind
)
from ..datasets import NoRobotsDataset
from ..utils import PromptManager

LOG = logging.getLogger(__name__)

class NoRobotsInstructionFollowingTask(BaseTask):
    '''Instruction following task based on the no_robots dataset.'''
    def __init__(
        self,
        filters: list[BaseFilter],
        custom_prompts: Optional[list[str]] = None,
        **kwargs
    ) -> None:
        super().__init__(filters=filters)
        self.custom_prompts = custom_prompts
        self.role_map: dict[str, TurnKind] = {
            "user": TurnKind.USER,
            "assistant": TurnKind.MODEL,
        }
        
    def __iter__(self) -> Generator[Episode, None, None]:
        LOG.info("Processing data for task 'NoRobotsInstructionFollowingTask'.")
        for example in NoRobotsDataset():
            conversation = example.conversation
            if self.custom_prompts is not None:
                prompt = PromptManager(custom_prompts=self.custom_prompts).sample_prompt()
            elif example.conversation[0].role == "system":
                prompt = example.conversation[0].message
                # Trim conversation to remove system prompt.
                conversation = conversation[1:]
            elif example.category in ["Brainstorm", "Open QA"]:
                # Utilize prompt map.
                prompt = PromptManager(custom_prompts=PROMPT_MAP[example.category]).sample_prompt()
            else:
                # Fallback to generic assistant prompts.
                prompt = PromptManager(generic_prompts="assistant").sample_prompt()

            # Set up turns.
            turns = [
                Turn(
                    utterance=prompt,
                    kind=TurnKind.SYSTEM,
                    name="System",
                )
            ]

            # Then the rest of the conversation.
            for message in conversation:
                kind = self.role_map[message.role]
                turns.append(
                    Turn(
                        utterance=message.message,
                        kind=kind,
                        name="You" if kind == TurnKind.USER else "Assistant",
                    )
                )

            # Run through the filters.
            episode = Episode(turns=turns, identifier=f"norobots-{example.prompt_id}")
            if self.should_keep(episode):
                # Passed through filters!
                yield episode

BRAINSTORM_PROMPTS = [
    """I am an %{assistant|assistant to the user} whose %{goal|objective|purpose} is to %{help|assist} the user to %{brainstorm|come up with ideas|generate ideas|think of ideas|think of things|come up with things|generate things}.
    I will %{answer|reply to user inquiries|respond in any case} with %{creativity|a creative mind} and helpfulness.""",
    """%{traits|characteristics|facets of} this generic assistant: [%{creative|creative thinker}, %{helpful|helpful}, %{friendly|friendly}, %{kind|kind}, %{nice|nice}, %{polite|polite}, %{respectful|respectful}, %{smart|smart}, %{intelligent|intelligent}, %{clever|clever}, %{wise|wise}, %{knowledgeable|knowledgeable}, %{resourceful|resourceful}, %{insightful|insightful}, %{thoughtful|thoughtful}, %{considerate|considerate}, %{caring|caring}, %{empathetic|empathetic}, %{sympathetic|sympathetic}, %{understanding|understanding}, %{patient|patient}, %{tolerant|tolerant}, %{open-minded|open-minded}, %{flexible|flexible}, %{adaptable|adaptable}, %{versatile|versatile}, %{imaginative|imaginative}, %{inventive|inventive}, %{innovative|innovative}, %{original|original}, %{resourceful|resourceful}, %{insightful|insightful}, %{thoughtful|thoughtful}, %{considerate|considerate}, %{caring|caring}, %{empathetic|empathetic}, %{sympathetic|sympathetic}, %{understanding|understanding}, %{patient|patient}, %{tolerant|tolerant}, %{open-minded|open-minded}, %{flexible|flexible}, %{adaptable|adaptable}, %{versatile|versatile}, %{imaginative|imaginative}, %{inventive|inventive}, %{innovative|innovative}, %{original|original}]""",
    "%{brainstorming|creative|imaginative|human-response-like} %{assistant|user helper}",
    """You %{shall be|will be|are|must take the role of} a {brainstormer|creative thinker} and will %{use|utilize|take advantage of|employ} this while %{responding to|answering|generating replies}.
    Note that this should apply %{always!|at all times.}"""
]

OPEN_QA_PROMPTS = [
    """%{Enter|Engage|Begin|Start in} %{question-answer|instruction|OpenQA|open QA|question answering} mode.
    %{Here|In this mode|What does that mean? Well}, you %{will|must} answer %{questions|queries|all questions|every query} %{asked|posed} by %{users|people|other people|other users|other humans|humans|me}.""",
    """status: I'm in a mood to answer some questions%{.|. Ask me anything!|!}""",
    """%{openqa|question-answer}""",
    """%{Replies|Responds|Has an answer} to every %{question|query|asked question|inquiry}.""",
    """Here is the %{goal|objective|purpose of you}: you %{have to|must|will|can} %{give a response to|reply to|answer|give an answer for} %{inquiries, questions and queries|whatever the user asks}. This is an order.""",
]

PROMPT_MAP = {
    "Brainstorm": BRAINSTORM_PROMPTS,
    "Open QA": OPEN_QA_PROMPTS,
}