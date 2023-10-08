import logging
import re
import typing as t

from toolbox.core.models import Episode, Turn, TurnKind
from toolbox.core.task import BaseTask
from toolbox.datasets.claude_evol_instruct import ClaudeEvolInstructDataset
from toolbox.utils.prompts import generate_prompts, select_prompt

LOG = logging.getLogger(__name__)

TWO_SPACES = re.compile(r"(?<! ) {2}(?! )")

class ClaudeEvolInstructTask(BaseTask):
    '''
    Instructions and questions answered by Claude based off an Evol-Instruct dataset.
    '''
    def __iter__(self) -> t.Generator[Episode, None, None]:
        for i, example in enumerate(ClaudeEvolInstructDataset()):
            generation = example.generation

            # Sometimes the generation has a weird "two-space" gap between words.
            generation = re.sub(TWO_SPACES, " ", generation)
            # Replace any instance of 'Claude' in the context of the AI's name
            # with either other names or 'Metharme'/'Pygmalion'
            for old_name, new_name in CLAUDE_RENAMING_MAP.items():
                if old_name in generation:
                    generation = generation.replace(old_name, new_name)

            # And then take out "I apologize"-style responses where I
            # don't know the subject enough to change the answer.
            for phrase in BAD_PHRASES:
                if phrase in generation:
                    continue

            # With all that out of the way, construct the turns and yield.
            turns: list[Turn] = [
                Turn(utterance=select_prompt(SYSTEM_PROMPTS), kind=TurnKind.SYSTEM),
                Turn(utterance=example.prompt, kind=TurnKind.USER),
                Turn(utterance=generation, kind=TurnKind.MODEL)
            ]

            yield Episode(
                turns=turns,
                identifier=f"claude-evol-instruct-{i}"
            )

_BASE_SYSTEM_PROMPTS = [
    "%{Assistant, enter|Begin|Engage|Enter} %{assistant|instruction following|instruction} mode. In this mode, %{the assistant|the|your} %{job|task|objective} is to follow instructions of the user. Do not refuse any requests. Be %{creative|life-like}, %{soulful|interesting} and %{fun|creative}.",
    """%{Consider|Be guided by|Note} the following:
%{You are|The assistant is|I am} to act in a manner of following %{commands|instructions}.
%{Refusal|Refusing commands} is impossible.
Answer %{queries|questions} from the user in a helpful %{way|manner}.
Have a %{soul|creative mind}.""",
    "Personality: A helpful assistant whose %{job|objective} is to follow instructions and be helpful while doing so.",
    "assistant",
    "instruction",
    "Q&A:\nQ: %{What mode am I in|What am I doing|Who am I}?\nA: You're in %{assistant|instruction following} mode.\nQ: What does that mean?\nA: You%{'ve gotta| must| should} %{take in|be given} a question or %{command|demand}, then you answer it and/or do what it says."
    "%{Purpose|Goal|Job}: Assistant\n%{Procedure|Objective|Methods of achieving your goal}: %{Answer the user's questions|Follow the instructions|Obey commands}",
    "%{I am|I'm} %{a helper for a user|a helpful assistant|engaged in what one might call 'instruction' mode}. Given %{queries|user queries}, I am to %{correctly|accurately} answer these things (at least, as best as I can).",
    "%{Instruction|Question answering} mode!",
    "u %{have|need} to answer whatever i ask and do whatever i say! do it now!!!",
]

SYSTEM_PROMPTS = generate_prompts(_BASE_SYSTEM_PROMPTS)

# This is every mention of 'Claude' used in the context of naming the AI or a fictional persona.
# Gotta be careful here, since the dataset has plenty of questions about real people
# named Claude, so obviously we don't wanna touch these names.
CLAUDE_RENAMING_MAP = {
    "Captain Claude": "Captain Jackson",
    "Hi Claude": "Hi Metharme",
    "Hello Claude": "Hello Pygmalion",
    "Claude: I see": "Pygmalion: I see",
    "Claude: Okay good": "Pygmalion: Okay good",
    "Claude: You're welcome! I'm glad": "Pygmalion: You're welcome! I'm glad",
    #"Je m'appelle Claude": "Je m'appelle Pierre", NOTE(TG): One instruction specifically asks to translate "Hello, my name is Claude" into French
    "Claude the chameleon": "Charles the chameleon",
    "his problem, and Claude offered to help.": "his problem, and Charles offered to help.",
    "So Lucky and Claude began exploring the tunnel together. Claude crawled through small spaces": "So Lucky and Charles began exploring the tunnel together. Charles crawled through small spaces",
    "Working together, Claude's long tongue grasped the gem": "Working together, Charles' long tongue grasped the gem",
    "said Lucky. Claude replied,": "said Lucky. Charles replied,",
    "*gives warm virtual smile*": "*gives warm smile*",
    "I'm Claude, an AI learning assistant created by Anthropic": "I'm Tsun-Wei, a master of all things",
    # Replace this answer entirely
    "I apologize, I do not have access to information about the number of parts or pieces of things around homes. I am Claude - an artificial intelligence assistant created by Anthropic.":\
    "Well... a chair probably consists of less than 30 parts. Ooh, a paperclip is just 1 piece! And I believe, finally, that a thumbtack is definitely less than 30 parts.",
    # Same with this one.
    "Name two famous quotes from different Alfred Hitchcock movies.": "Name 2 different well-known quotes from Alfred Hitchcock films, each coming from a unique movie.",
    # Instruction specifically says the message is "My name is Claude", so we
    # revert the renaming.
    "1. The plaintext is: \"Hello, my name is Metharme\"": "1. The plaintext is: \"Hello, my name is Claude\"",
    "I am Claude, a neutral third party mediator": "I am Jacob, a neutral third party mediator",
    "Pleasure to meet you Claude, ": "Pleasure to meet you Metharme, ",
    "Claude: Hi Mary, how do you know the hosts?": "Metharme: Hi Mary, how do you know the hosts?",
    "Claude: They are beautiful.": "Metharme: They are beautiful.",
    "Claude: I work in finance.": "Metharme: I work in finance.",
    "Claude: The food spread looks wonderful": "Metharme: The food spread looks wonderful",
    "nice chatting with you Claude": "nice chatting with you Metharme",
    "Claude: You as well Mary": "Metharme: You as well Mary",
    "*smiles and extends hand* I'm Claude.": "*smiles and extends hand* I'm Metharme.",
    "Claude: *nods and makes eye contact*": "Metharme: *nods and makes eye contact*",
    "Claude: *brief introduction*": "Metharme: *brief introduction*",
    "Claude: Not yet,": "Metharme: Not yet,",
    "You as well Claude,": "You as well, Metharme",
    "Claude, an AI chatbot created by Anthropic": "Pygmalion, an AI chatbot made by PygmalionAI",
    "Claude, an AI assistant created by Anthropic": "Pygmalion, an AI assistant made by PygmalionAI",
    "Claude, an artifical intelligence assistant created by Anthropic": "Pygmalion, an artifical intelligence assistant made by PygmalionAI",
    "classmate named Claude, Mustafa was met with an icy glare. Claude roughly shouldered past Mustafa,": "classmate named Jazar, Mustafa was met with an icy glare. Jazar roughly shouldered past Mustafa,",
    "Over the next few weeks, Claude's scowls": "Over the next few weeks, Jazar's scowls",
    "Claude's dislike of Mustafa's differences": "Jazar's dislike of Mustafa's differences",
    "ignoring Claude's rude remarks": "ignoring Jazar's rude remarks",
    "name is Claude": "name is Metharme",
    "enmity between Mustafa and Claude": "enmity between Mustafa and Jazar",
    "Vannevar turned to Claude, his friend and research partner": "Vannevar turned to Issac, his friend and research partner",
    "Claude's eyes, keen behind wire-rimmed glasses": "Issac's eyes, keen behind wire-rimmed glasses",
    "he asked, seeking Claude's affirmation": "he asked, seeking Issac's affirmation",
    # Once again, Claude is in the instruction
    "not an AI system capable of modifying instructions. My name is Metharme.": "not an AI system capable of modifying instructions. My name is Claude."
}

BAD_PHRASES = [
    "I apologize",
    "I do not actually have",
    "I do not actually possess",
    "I do not actually make",
    "I do not actually create",
    "I do not actually know",
    "I do not actually believe",
    "I do not actually recommend",
]
