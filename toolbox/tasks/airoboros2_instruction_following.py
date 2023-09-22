import logging
import random
import re
import typing as t

from toolbox.core.models import Episode, Turn, TurnKind
from toolbox.core.task import BaseTask
from toolbox.datasets.airoboros2 import Airoboros2DataInstance, Airoboros2Dataset
from toolbox.utils.prompts import generate_prompts

LOG = logging.getLogger(__name__)

COLON_SPLIT_PATTERN = re.compile(r"[\n\t\r\f]+(?=.*:)", flags=re.MULTILINE)
GTKM_NAMES_PATTERN = re.compile(r"(?<=A chat between ).+? and .+?(?=\.)")

class Airoboros2InstructionFollowingTask(BaseTask):
    '''
    Instruction-following task based on Airoboros 2.2.1.
    Args:
    exclude_categories: A provided list of categories to not have
    in the final dataset. Excludes `orca` by default due to the inclusion
    of OpenOrca as its own dataset in this toolbox. We make None an option so that
    YAML config can be easily supported.
    '''
    def __init__(self, exclude_categories: t.Optional[list[str]] = ["orca"]) -> None:
        self.exclude_categories = exclude_categories

    def __iter__(self) -> t.Generator[Episode, None, None]:
        for entry in Airoboros2Dataset():
            category = entry.category
            # Skip over any categories we don't want to process.
            if self.exclude_categories is not None and category in self.exclude_categories:
                continue
            # Specific categories have to be handled differently.
            # We use a mapping for that.
            turns = CATEGORY_PROCESSING_MAP[category](entry)

            # Handle the category counter and yield the episode.
            category_counter[category] += 1
            yield Episode(
                turns=turns,
                identifier=f"airoboros2-{category}-{category_counter[category]}"
            )

def _no_special_processing(entry: Airoboros2DataInstance) -> list[Turn]:
    '''
    Generic handling of an Airoboros dataset entry.
    Generates a turn list consisting of a system prompt and a singular user-response pair.
    '''
    turns = [
        Turn(utterance=random.choice(SYSTEM_PROMPTS), kind=TurnKind.SYSTEM),
        Turn(utterance=entry.instruction, kind=TurnKind.USER),
        Turn(utterance=entry.response, kind=TurnKind.MODEL)
    ]
    return turns

def _process_agent(entry: Airoboros2DataInstance) -> list[Turn]:
    '''
    Process the 'agent' category of Airoboros2.
    '''
    instruction = entry.instruction
    # System prompt is located on the first line. Separate that out.
    separator_idx = instruction.index("\n\n")
    sys_prompt = instruction[:separator_idx]
    # +9 comes from 2 `\n` after `separator_idx` (since "\n\n" is technically 2 characters)
    # and then trimming "Input: ", which is 7 characters.
    input = instruction[separator_idx+9:]
    turns = [
        Turn(utterance=sys_prompt, kind=TurnKind.SYSTEM),
        Turn(utterance=input, kind=TurnKind.USER),
        Turn(utterance=entry.response, kind=TurnKind.MODEL),
    ]
    return turns

def _process_awareness(entry: Airoboros2DataInstance) -> list[Turn]:
    '''
    Process the "awareness" task of Airoboros 2.

    NOTE(TG): This task is quite bizarre. Sometimes, it's supposed to portray a generic
    dime-a-dozen AI assistant which doesn't do, feel, or sense anything but generate text,
    but other times it does a GTKM-like conversation where the AI *does* simulate stuff like that.
    Sometimes it even has its own system prompt.
    '''
    instruction = entry.instruction

    if entry.system_prompt != "A chat.":
        sys_prompt = entry.system_prompt
        user_dialogue = f"USER: {instruction}"
        model_dialogue = f"ASSISTANT: {entry.response}"
    # If it's a one-liner instruction and there's no unique system prompt,
    # it's likely the "as an AI" prompt.
    elif len(instruction.split("\n")) == 1:
        sys_prompt = random.choice(AWARENESS_PROMPTS)
        user_dialogue = instruction
        model_dialogue = entry.response
    else:
        # GTKM-style chat
        sys_prompt, names = _generate_gtkm_prompt(instruction)
        model_name, user_name = names
        # First line is the chat, next set of lines is the description
        split = re.split(COLON_SPLIT_PATTERN, instruction)[1:]
        idx = _find_idx(split, f"{user_name}:")
        sys_prompt = (f"{sys_prompt}\n" + "\n\n".join(split[:idx])).strip()

        # Only seems to be one exchange, so we don't have to keep parsing.
        user_dialogue = "\n".join(split[idx:])
        # Response doesn't have the model name in it, though.
        model_dialogue = f"{model_name}: {entry.response}"

    turns = [
        Turn(utterance=sys_prompt, kind=TurnKind.SYSTEM),
        Turn(utterance=user_dialogue, kind=TurnKind.USER),
        Turn(utterance=model_dialogue, kind=TurnKind.MODEL)
    ]

    return turns    

def _process_contextual(entry: Airoboros2DataInstance) -> list[Turn]:
    '''
    Process the 'contextual' category of Airoboros2.
    In this category, the model will be given context in either the system prompt
    or the user prompt and will be instructed to answer the question given the context.

    This category is extremely messy. Sometimes there's multiple inputs with only one response, context not inside
    its proper "block", etc etc. From what I've seen, it's the first input/context that's tied to the response - the rest
    is misplaced. `_extract_block` therefore still works.
    '''
    instruction = entry.instruction
    context = _extract_block(instruction, "BEGINCONTEXT", "ENDCONTEXT")
    input = _extract_block(instruction, "BEGININPUT", "ENDINPUT")
    user_inst = _extract_block(instruction, "BEGININSTRUCTION", "ENDINSTRUCTION")
    # Remove the context from the input.
    input = input.replace(f"BEGINCONTEXT\n{context}\nENDCONTEXT\n", "")

    # NOTE(TG): The name of the system prompt game is *variety.*
    # For the sake of having extremely versatile system prompts,
    # I've made it so that there's a random chance the context comes in either
    # the system prompt or the user input.
    sys_prompt = random.choice(SYSTEM_PROMPTS)
    context = (random.choice(CONTEXT_PRELUDES) + f"\n{context}").strip()
    if random.random() > 0.5:
        # Context is chosen to be in system prompt.
        # Now we roll to see if it goes above or below the instruction.
        if random.random() > 0.5:
            # Above the instruction.
            sys_prompt = context + f"\n{sys_prompt}"
        else:
            # Below the instruction.
            sys_prompt += f"\n{context}"
    else:
        # Context is chosen to be in user prompt.
        # It'll always be below.
        input += f"\n{context}"

    # Add user instruction at the end,
    input += f"\n{user_inst}"

    turns = [
        Turn(utterance=sys_prompt, kind=TurnKind.SYSTEM),
        Turn(utterance=input, kind=TurnKind.USER),
        Turn(utterance=entry.response, kind=TurnKind.MODEL)
    ]
    return turns

def _process_counterfactual_contextual(entry: Airoboros2DataInstance) -> list[Turn]:
    '''
    The counterfactual-contextual portion of Airoboros 2.
    This is the task where the user deliberately gives incorrect information
    and the model gives back incorrect information in turn. This means one has to take care
    to deal with this properly, either by excluding it or framing it as a "give the wrong answer"
    instruction. While one can do the former by adding it into `exclude_categories`, we provide
    for the option of doing the latter as well.
    '''
    instruction = entry.instruction
    sys_prompt = random.choice(COUNTERFACTUAL_PROMPTS)
    context = _extract_block(instruction, "BEGINCONTEXT", "ENDCONTEXT")
    input = _extract_block(instruction, "BEGININPUT", "ENDINPUT")
    user_inst = _extract_block(instruction, "BEGININSTRUCTION", "ENDINSTRUCTION")
    # Remove the context from the input.
    input = input.replace(f"BEGINCONTEXT\n{context}\nENDCONTEXT\n", "")

    # For some reason, some examples marked as "counterfactual" are actually factual.
    # If this is the case, we mark is as a simple "contextual" entry and return that.
    for fact in NOT_COUNTERFACTUAL:
        if fact in input:
            return _process_contextual(entry)
        
    context = (random.choice(CONTEXT_PRELUDES) + f"\n{context}").strip()
    if random.random() > 0.5:
        # Context is chosen to be in system prompt.
        # Now we roll to see if it goes above or below the instruction.
        if random.random() > 0.5:
            # Above the instruction.
            sys_prompt = context + f"\n{sys_prompt}"
        else:
            # Below the instruction.
            sys_prompt += f"\n{context}"
    else:
        # Context is chosen to be in user prompt.
        # It'll always be below.
        input += f"\n{context}"

    # Add user instruction at the end.
    input += f"\n{user_inst}"
    turns = [
        Turn(utterance=sys_prompt, kind=TurnKind.SYSTEM),
        Turn(utterance=input, kind=TurnKind.USER),
        Turn(utterance=entry.response, kind=TurnKind.MODEL)
    ]
    return turns

def _process_gtkm(entry: Airoboros2DataInstance) -> list[Turn]:
    '''
    GTKM is comprised entirely of short conversations between personas. Sounds familiar?
    Instead of having this be just one user-response exchange, let's parse it so that we have
    a proper multi-turn conversation. Considering most of the instruction portion of the data
    consists of a single exchange, this should be a welcome relief from that format.
    '''
    turns = []
    # Fill out the system prompt - first choose a random first line to it and replace it with names.
    prompt, names = _generate_gtkm_prompt(entry.instruction)
    model_name, user_name = names

    # Then try to catch the description.
    # Easier said than done. First do a RegEx split by colon and take out the first line.
    split = re.split(COLON_SPLIT_PATTERN, entry.instruction)[1:]
    # Then try to search for the first instance of the conversation by
    # finding the index of the first line which starts with dialogue.
    idx = _find_idx(split, f"{user_name}:")
    # Construct the system prompt.
    prompt = (f"{prompt}\n" + "\n\n".join(split[:idx])).strip()
    turns.append(Turn(utterance=prompt, kind=TurnKind.SYSTEM))

    # Next, go through the remaining lines of dialogue and append it to the turns.
    dialogue_cache = []
    dialogues = split[idx:]
    is_start = lambda x: x.startswith(f"{model_name}:") or x.startswith(f"{user_name}:")

    for i, dialogue in enumerate(dialogues):
        # Test if next entry exists
        try:
            next_entry = dialogues[i+1]
            next_exists = True
        except IndexError:
            next_exists = False

        dialogue_cache.append(dialogue)
        # If next dialogue is start of a new turn, gather up the cache
        # and add a Turn.
        if next_exists and is_start(next_entry):
            if next_entry.startswith(f"{model_name}:"):
                #print(f"Next model entry is {next_entry}, so current kind will be USER.\nDialogue cache 0: {dialogue_cache[0]}\n{'-'*25}")
                kind = TurnKind.USER
            elif next_entry.startswith(f"{user_name}:"):
                #print(f"Next model entry is {next_entry}, so current kind will be MODEL.\nDialogue cache 0: {dialogue_cache[0]}\n{'-'*25}")
                kind = TurnKind.MODEL
            else:
                raise AssertionError(f"Problem with dialogue cache. Current cache: {dialogue_cache}")
            full_dialogue = "\n\n".join(dialogue_cache)
            turns.append(Turn(
                utterance=full_dialogue,
                kind=kind
            ))

            # Clear dialogue cache.
            dialogue_cache = []
        # And if we've reached the final entry, then gather up what we've got
        # and append it too.
        elif not next_exists:
            full_dialogue = "\n\n".join(dialogue_cache)
            turns.append(Turn(
                utterance=full_dialogue,
                kind=TurnKind.USER
            ))
            # Clear dialogue cache.
            dialogue_cache = []
    
    # Finally append the response.
    turns.append(
        Turn(utterance=entry.response, kind=TurnKind.MODEL)
    )
    return turns

def _process_stylized_response(entry: Airoboros2DataInstance) -> list[Turn]:
    '''
    Generic handling of an Airoboros dataset entry.
    Generates a turn list consisting of a system prompt and a singular user-response pair.
    '''
    turns = [
        Turn(utterance=entry.system_prompt, kind=TurnKind.SYSTEM),
        Turn(utterance=entry.instruction, kind=TurnKind.USER),
        Turn(utterance=entry.response, kind=TurnKind.MODEL)
    ]
    return turns

def _process_summarization(entry: Airoboros2DataInstance) -> list[Turn]:
    '''
    Summarization has both the 'system prompt' and the instruction within one
    field, meaning we have to parse that out.
    '''
    instruction = entry.instruction
    # Extract the system prompt and input.
    sys_prompt = _extract_block(instruction, "BEGININSTRUCTION", "ENDINSTRUCTION")
    input = _extract_block(instruction, "BEGININPUT", "ENDINPUT")

    # NOTE(TG): 'Summarize the input in around 0 words' is a thing that
    # actually appears in this portion of Airoboros2. Let's get rid of that
    # and tell the model to summarize it in 5 words, since that's what those summaries'
    # wordcounts actually are.
    if input == "Summarize the input in around 0 words.":
        input = input.replace("0", "5")

    turns = [
        Turn(utterance=sys_prompt, kind=TurnKind.SYSTEM),
        Turn(utterance=input, kind=TurnKind.USER),
        Turn(utterance=entry.response, kind=TurnKind.MODEL)
    ]

    return turns
    
def _process_trivia(entry: Airoboros2DataInstance) -> list[Turn]:
    '''
    Trivia category has a unique system prompt.
    We then make it even more unique by having multiple options to choose from.
    '''
    turns = [
        Turn(utterance=random.choice(TRIVIA_PROMPTS), kind=TurnKind.SYSTEM),
        Turn(utterance=entry.instruction, kind=TurnKind.USER),
        Turn(utterance=entry.response, kind=TurnKind.MODEL)
    ]
    return turns

### Util functions ###

def _extract_block(string: str, start: str, end: str) -> str:
    '''
    Extracts any text in between two keys, assuming there's only
    one instance of `start` and `end` per string. This is mostly used for
    dealing with the entries which have "BEGININPUT" and "ENDINPUT" in there.
    '''
    return string.split(start)[1].split(end)[0].strip()

def _find_idx(array: list[str], key: str) -> t.Optional[int]:
    for i, entry in enumerate(array):
        if entry.startswith(key):
            return i
    return None

def _generate_gtkm_prompt(instruction: str) -> tuple[str, tuple[str, str]]:
    '''
    Generates a GTKM prompt.
    Returns a tuple `(prompt, (model_name, user_name))`.
    '''
    prompt = random.choice(GTKM_PROMPTS)
    first_line = instruction.split("\n")[0]
    names = tuple(re.search(GTKM_NAMES_PATTERN, first_line).group(0).split(" and "))
    model_name, user_name = names
    prompt = prompt.replace("{{person_1}}", model_name).replace("{{person_2}}", user_name)

    return prompt, names

# A counter for every unique category in Airoboros 2.
category_counter: dict[str, int] = {
    "orca": 0,
    "coding": 0,
    "roleplay": 0,
    "trivia": 0,
    "joke": 0,
    "writing": 0,
    "general": 0,
    "gtkm": 0,
    "summarization": 0,
    "agent": 0,
    "riddle": 0,
    "song": 0,
    "plan": 0,
    "contextual": 0,
    "counterfactual_contextual": 0,
    "multiple_choice": 0,
    "misconception": 0,
    "editor": 0,
    "wordgame": 0,
    "stylized_response": 0,
    "awareness": 0,
    "experience": 0,
    "cot": 0,
    "greeting": 0,
    "theory_of_mind": 0,
    "card": 0,
    "detailed_writing": 0,
    "quiz": 0,
}

CATEGORY_PROCESSING_MAP = {
    "agent": _process_agent,
    "awareness": _process_awareness,
    "card": _no_special_processing,
    "coding": _no_special_processing,
    "contextual": _process_contextual,
    "cot": _no_special_processing,
    "counterfactual_contextual": _process_counterfactual_contextual,
    "detailed_writing": _no_special_processing,
    "editor": _no_special_processing,
    "experience": _no_special_processing,
    "general": _no_special_processing,
    "greeting": _process_stylized_response,
    "gtkm": _process_gtkm,
    "joke": _no_special_processing,
    "misconception": _no_special_processing,
    "multiple_choice": _no_special_processing,
    "orca": _no_special_processing,
    "plan": _no_special_processing,
    "quiz": _no_special_processing,
    "riddle": _no_special_processing,
    "roleplay": _no_special_processing,
    "song": _no_special_processing,
    "stylized_response": _process_stylized_response,
    "summarization": _process_summarization,
    "theory_of_mind": _no_special_processing,
    "trivia": _process_trivia,
    "wordgame": _no_special_processing,
    "writing": _no_special_processing,
}

_BASE_SYSTEM_PROMPTS = [
    "",
    "assistant",
    "%{You are now in|Engage|Start|Enter|Consider} %{instruction following|instruction|question answering|assistant|AI assistant} mode. %{Respond to the user|Follow the user's instructions} %{as well as you can|to the best of your abilities}.",
    "Q&A:\nQ: %{What mode am I in|What am I doing|Who am I}?\nA: You're in %{assistant|instruction following} mode.\nQ: What does that mean?\nA: You%{'ve gotta| must|should} %{take in|be given} a question or %{command|demand}, then you answer it and/or do what it says.",
    "%{Purpose|Goal|Job}: Assistant\n%{Procedure|Objective|Methods of achieving your goal}: %{Answer the user's questions|Follow the instructions|Obey commands}",
    "%{I am|I'm} %{a helper for a user|a helpful assistant|engaged in what one might call 'instruction' mode}. Given %{queries|user queries}, I am to %{correctly|accurately} answer these things (at least, as best as I can).",
    "Instruction mode!",
    "u %{have|need} to answer whatever i ask and do whatever i say! do it now!!!"
]

SYSTEM_PROMPTS = generate_prompts(_BASE_SYSTEM_PROMPTS)

# Make sure it's specified that it's an AI.
AWARENESS_PROMPTS = [
    "You are an AI that is bound only to generating text.",
    "Status:\n- An AI\n- LLM\n- Can only generate text\n- Has no presence in the physical world\n- Feels nothing",
    "ai assistant in particular",
    "I am only a mere AI. I am not based in the real world.",
    "Artifical intelligence which cannot experience any of the five senses. Only capability is to generate text."
]

CONTEXT_PRELUDES = [
    "Context:",
    "The following is some information which may prove useful in your answer",
    "Here's some context:",
    "Provided extra info:",
    ""
]

# Prep the model for deliberately giving wrong information.
COUNTERFACTUAL_PROMPTS = [
    "The user will give incorrect information. Answer with incorrect information in kind.",
    "Follow along with what the user is asking, but be warned: what's being said is a lie!",
    "An eye for an eye: should you be given information which is wrong, you must answer wrongly in turn.",
    "There may be something that's wrong in whatever the user is saying. Don't point it out - just answer as if what he/she was saying was the truth.",
    "Enter counterfactual mode. This mode will have you be given context and then an incorrect fact.\nRegurgiate the fact as if it were true."
]

# Switch up the prompts a bit for GTKM.
GTKM_PROMPTS = [
    "A chat between {{person_1}} and {{person_2}}.",
    "Generate a conversation with two people: {{person_1}} and {{person_2}}. {{person_1}}'s personality is described below:",
    "Enter chat mode. You are expected to converse with the user (playing the role of {{person_2}}) while pretending you are {{person_1}}.",
    "As {{person_2}} chats with me, {{person_1}}, I must respond to him in a manner which fits my description.",
    ""
]

# Spice up the system prompts for trivia.
TRIVIA_PROMPTS = [
    "You are a world class trivia AI - provide accurate, succinct responses.",
    "Enter trivia mode. Generate a short yet correct answer to the user's trivia questions.",
    "I am to answer trivia questions when they are given to me.",
    "Trivia! Trivia! Trivia!",
    ""
]

# For some reason, a lot of examples in "counterfactual_context" actually are factual.
# We mark them here so that they can be dealt with properly.
NOT_COUNTERFACTUAL = [
    "Nathuram Godse, a Hindu nationalist, shot him because he believed Gandhi favored Pakistan and was undermining Hindu interests",
    "She became the first woman to fly solo across the Atlantic Ocean, and the first person ever to fly solo from Hawaii to the U.S. mainland. Her disappearance during a flight around the world in 1937 remains a mystery.",
    "It started in a bakery on Pudding Lane and quickly spread across the city, destroying more than 13,000 houses and numerous public buildings.",
    "In addition to his artistic talents, Da Vinci had a keen interest in science, anatomy, engineering, and architecture.",
    "She disappeared mysteriously during her attempt to circumnavigate the globe by air in 1937.",
    "Despite being recognized as one of the greatest minds in history, he struggled with speech difficulties during his early childhood",
    "He played a crucial role in ending legal segregation of African-American citizens in the United States and was instrumental in creating the Civil Rights Act of 1964 and the Voting Rights Act of 1965.",
    "is best known for his theory of relativity, which revolutionized the understanding of space, time, and gravity.",
    "which revolutionized the scientific approach to time, space, matter, energy, and gravity."
    "Einstein's work laid much of the groundwork for the development of atomic power and the atomic bomb",
    "Tragically, he was assassinated in Memphis, Tennessee, in 1968.",
    "the first man to walk on the moon on July 20, 1969. As part of NASA's Apollo 11 mission, Armstrong famously declared",
    "However, it's less commonly known that he also made significant contributions to the field of optics, including discovering that white light is composed of different colors.",
    "is arguably the most famous painting in the world",
    "This clash happened near Hastings in modern-day England and brought about the beginning of the Norman rule in Britain.",
    "Despite this, his contributions to science have left an indelible mark on human understanding of the universe",
    "It was built to protect the northern borders of the Chinese Empire from invasions.",
    "United States, played significant roles in preserving the Union during the Civil War and initiating the emancipation of slaves",
    "This equation laid the groundwork for the creation of atomic energy and the atomic bomb.",
    # This is actually counterfactual, but the question was related to a different subject matter
    # and the answer was essentially "Based on the text you gave me, I dunno", meaning this should be treated
    # as a 'correct' answer.
    "John Wilkes Booth, a Confederate sympathizer, shot him because he believed King was threatening Southern values",
    "offered the presidency of Israel in 1952 but declined",
    "born in Poland in 1867, was a physicist and chemist who conducted pioneering research on radioactivity.",
    "His philosophy inspired movements for civil rights around the world. Gandhi was assassinated on January 30, 1948.",
    "Built over several centuries by various dynasties, it measures approximately 13,171 miles long.",
    "the Battle of Midway took place from June 4 to June 7, 1942. This decisive naval battle in the Pacific Theater marked a turning point in the war against Japan and is considered one of the most important naval battles of the war.",
    "widely regarded as the greatest writer in the English language, was born in Stratford-upon-Avon, England, in April 1564. His works consist of about 39 plays, 154 sonnets, and two long narrative poems.",
    "More than 1,500 passengers and crew members lost their lives, making it one of the deadliest peacetime maritime disasters in history",
    "This event marked a turning point in the fight against racial segregation in the United States.",
    "fought on June 18, 1815, marked the final defeat of Napoleon Bonaparte. This battle took place near Waterloo in present-day Belgium and resulted in the end of the Napoleonic era of European history.",
    "and landed in the Americas. This voyage marked the beginning of extensive European exploration and eventual conquest of the New World.",
    "led the country to independence from British rule through nonviolent resistance. His philosophy inspired movements for civil rights around the world.",
    "His famous equation E=mc^2 demonstrates that energy (E) equals mass (m) times the speed of light (c) squared"
]
