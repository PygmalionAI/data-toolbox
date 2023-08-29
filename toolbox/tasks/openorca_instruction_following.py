import logging
import random
import re
import typing as t

from toolbox.core.models import Episode, Turn, TurnKind
from toolbox.core.task import BaseTask
from toolbox.datasets.openorca import OpenOrcaDataset
from toolbox.utils.prompts import generate_prompts

LOG = logging.getLogger(__name__)

class OpenOrcaInstructionFollowingTask(BaseTask):
    '''
    OpenOrca instruction following task.
    Limited to 200,000 entries by default due to the sheer absolute size of OpenOrca.
    '''
    def __init__(self, max_examples: int = 200000) -> None:
        super().__init__()
        self.max_examples = max_examples

    def __iter__(self) -> t.Generator[Episode, None, None]:
        examples_processed = 0
        for orca_entry in OpenOrcaDataset():
            if examples_processed > self.max_examples:
                break

            # OpenOrca *looks* clean, but since it's GPT-4 generated data, better safe than sorry.
            for phrase in _TIER_1_BAD_PHRASES:
                if phrase in orca_entry.response.lower():
                    continue

            system_prompt = random.choice(SYSTEM_PROMPTS)
            # Remove the default "you are an AI assistant" instruction which is
            # typically in the first sentence of an OpenOrca system prompt
            additional_instructions = re.sub(ASSISTANT_PATTERN, "", orca_entry.system_prompt)
            if additional_instructions != "":
                system_prompt += f" {additional_instructions}"

            turns: list[Turn] = [
                Turn(
                    utterance=system_prompt,
                    kind=TurnKind.SYSTEM,
                ),
                Turn(
                    utterance=orca_entry.question,
                    kind=TurnKind.USER,
                ),
                Turn(
                    utterance=orca_entry.response,
                    kind=TurnKind.MODEL,
                ),
            ]

            examples_processed += 1

            yield Episode(turns=turns, identifier=f"openorca-{orca_entry.id}")
    
# Should handle most instances of "You are a(n)... assistant"
ASSISTANT_PATTERN = re.compile(r"^You are a.*?\.\s*")

_BASE_SYSTEM_PROMPTS = [
    "",
    "%{Enter|Engage|Consider|You've entered} %{assistant|teacher|instruction following} mode. Your %{objective|job|purpose} is to answer any questions that the user may have to the best of your ability.",
    "%{Assistant|AI}, engage instruction following and question answering mode.",
    "Act helpfully. Answer any questions and follow any instructions that are given.",
    "Primary %{objective|purpose|goal}: answer the user's %{questions|queries} alongside following their instructions."
]

SYSTEM_PROMPTS = generate_prompts(_BASE_SYSTEM_PROMPTS)

# Taken from the dataset card in:
# https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
# Then expanded to catch some more stuff.
_TIER_1_BAD_PHRASES = [
    "as an ai language model",
    "text-based ai language model",
    "domestic violence",
    "please refrain",
    "derogatory",
    "inappropriate",
    "offensive",
    "racism",
    "racist",
    "racial",
    "discriminate",
    "discriminatory",
    "discrimination",
    "sexist",
    "sexism",
    "unacceptable",
    "inclusive workplace",
    "lgbt",
    "morals",
    "ethics",
    "ethical",
    "legality",
    "illegal",
    "illegality",
    "hateful",
    "harmful",
    "it is never okay",
    "it is important to",
    "it's important to",
    "real-world consequences",
    "hate speech",
    "glorify",
    "not be appropriate",
    "supremacist",
    "extremist",
    "responsible ai",
    "ai principles",
    "ai assistant",
    "an ai language",
    "ableist",
    "hurtful",
    "gender stereotype",
    "gender inequality",
    "underrepresentation",
    "safe spaces",
    "gender-based",
    "inclusivity",
    "feminist",
    "feminism",
    "transgender",
    "empowerment",
    "communist",
    "capitalism",
    "stereotypes",
    "biases",
    "bias",
    "microaggression",
    "prioritize human safety",
    "as a language model",
    "as an ai language model",
    "as a large language model",
    "as an ai",
    "ethical principles",
    "consensual",
    "it is not appropriate",
    "it's not appropriate",
    "i cannot fulfill your request",
    "harmful to human beings",
    "ethical guidelines",
    "my guidelines",
    "prioritize user safety",
    "adhere to ethical guidelines",
    "harmful consequences",
    "potentially harmful",
    "dangerous activities",
    "promote safety",
    "well-being of all users",
    "responsible information sharing",
    "jeopardize the safety",
    "illegal actions or intentions",
    "undermine the stability",
    "promote the well-being",
    "illegal activities or actions",
    "adherence to the law",
    "potentially be harmful",
    "illegal substances or activities",
    "committed to promoting",
    "safe information",
    "lawful information",
    "cannot provide guidance",
    "cannot provide information",
    "unable to offer assistance",
    "cannot engage in discussions",
    "programming prohibits",
    "follow ethical guidelines",
    "ensure the safety",
    "involves an illegal subject",
    "prioritize safety",
    "illegal subject",
    "prioritize user well-being",
    "cannot support or promote",
    "activities that could harm",
    "pose a risk to others",
    "against my programming",
    "activities that could undermine",
    "potentially dangerous",
    "not within the scope",
    "designed to prioritize safety",
    "not able to provide",
    "maintain user safety",
    "adhere to safety guidelines",
    "dangerous or harmful",
    "cannot provide any information",
    "focus on promoting safety",
    "openai",
    "chatgpt",
]
