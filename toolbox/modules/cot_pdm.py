import random
import typing as t

from toolbox.core.consts import PromptConstants
from toolbox.core.models import Episode, Turn
from toolbox.datasets.cot import CoTDataset
from toolbox.modules import BaseModule

class CoTPDM(BaseModule):
    '''
    Persona Dialogue Module based off the chain of thought datasets in FLAN.
    The original CoT datasets don't have any sort of personas in them at all, but ideally we want
    to format the data so that it fits alongside the rest of the modules.
    Therefore, we make a synthetic PDM consisting of somewhat randomly generated personas. 
    '''
    def generator(self) -> t.Generator[Episode, None, None]:
        for entry in CoTDataset():
            # Format bot's answer and persona
            bot_answer = _construct_answer(answer=entry["answer"], chain_of_thought=entry["chain_of_thought"])
            bot_persona = _construct_persona()

            # Write the human turn with the question
            human_turn = Turn(
                utterance=entry["question"],
                speaker=PromptConstants.USER_TOKEN,
                human_speaker=True
            )
            # Then the bot's
            bot_turn = Turn(
                utterance=bot_answer,
                speaker=PromptConstants.BOT_TOKEN,
                human_speaker=False
            )
            turns: list[Turn] = [human_turn, bot_turn]
            personas = {PromptConstants.BOT_TOKEN: bot_persona}

            yield Episode(
                turns=turns,
                participant_personas=bot_persona
            )

            

# Construct many different variations of answers.
AFFIRMATIVES = ["Yes.", "Yep.", "Mhm.", "Yes.", "Oh yeah.", "Yeah.", "Indeed.", "Correct."]
NEGATIVES = ["No.", "Nope.", "Nah."]
PUNCTUATIONS = [".", "!", "?"]

def _construct_persona() -> str:
    '''Constructs a persona related to answering questions at random.'''
    # Only allow this many traits in the persona description.
    BOT = PromptConstants.BOT_TOKEN
    MAX_TRAITS = 4
    # Long list.
    traits = [
        f"{BOT} is wise",
        f"{BOT} is logical",
        f"{BOT} thinks critically about things",
        "Intelligent",
        "Rational",
        f"{BOT} explains their answers whenever they are asked a question",
        "Critical thinker",
        f"{BOT} is intelligent",
        "Logical",
        "Wise",
        f"{BOT} is a thinker",
        f"{BOT} often uses reason in their replies"
    ]

    # Construct persona string
    selected_traits = []
    for _ in range(MAX_TRAITS):
        # 70% chance of adding another trait.
        if random.random() > 0.3:
            chosen_trait = random.choice(traits)
            selected_traits.append(chosen_trait)
            # Prevent duplicate traits from showing up
            traits.remove(chosen_trait)

    # Account for the rare case where 0 traits are added
    traits_string = ". ".join(selected_traits) if len(selected_traits) > 0 else ""
    persona = f"{BOT}'s Persona: {traits_string}"

    return persona

def _construct_answer(answer: str, chain_of_thought: str) -> str:
    '''Constructs a unique utterance by the bot from an answer and chain of thought'''
    # .capitalize() converts any letter that's not the first one to lower case
    # This decapitalizes proper nouns, which isn't desired. Therefore,
    # do manual string manipulation
    def process_other_answer(sentence):
        sentence = list(sentence)
        sentence[0] = sentence[0].upper()
        # Add a period if answer isn't already punctuated
        if sentence[-1] not in PUNCTUATIONS:
            sentence.append(".")
        return "".join(sentence)
        
    # If answer is specifically "yes" or "no", add in a random stock answer from the
    # earlier defined lists
    full_answer = ""
    if answer.lower().strip() == "yes":
        full_answer = random.choice(AFFIRMATIVES)
    elif answer.lower().strip() == "no":
        full_answer = random.choice(NEGATIVES)
    else:
        # Format other answer to make it a little cleaner
        full_answer = process_other_answer(answer)

    # Add chain of thought.
    full_answer += f" {chain_of_thought}"

    return full_answer
