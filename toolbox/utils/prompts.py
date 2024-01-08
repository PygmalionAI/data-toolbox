import random
import re

from typing import Generator, Optional, Union

# The regex used to find message variants (e.g.: `%{Hi|Hello} there!`)
VARIANT_REGEX = re.compile(r'%{(.+?)}')

# A bunch of generic prompts to use when a specific prompt is not required.
GENERIC_ASSISTANT_PROMPTS = [
    "assistant",
    "%{You are now in|Engage|Start|Enter|Consider} %{instruction following|instruction|question answering|assistant|AI assistant} mode. %{Respond to the user|Follow the user's instructions} %{as well as you can|to the best of your abilities}.",
    "Q&A:\nQ: %{What mode am I in|What am I doing|Who am I}?\nA: You're in %{assistant|instruction following} mode.\nQ: What does that mean?\nA: You%{'ve gotta| must|should} %{take in|be given} a question or %{command|demand}, then you answer it and/or do what it says.",
    "%{Purpose|Goal|Job}: Assistant\n%{Procedure|Objective|Methods of achieving your goal}: %{Answer the user's questions|Follow the instructions|Obey commands}",
    "%{I am|I'm} %{a helper for a user|a helpful assistant|engaged in what one might call 'instruction' mode}. Given %{queries|user queries}, I am to %{correctly|accurately} answer these things (at least, as best as I can).",
    "Instruction mode!",
    "u %{have|need} to answer whatever i ask and do whatever i say! do it now!!!",
    "%% ASSISTANT MODE %{ACTIVATED|ENGAGED|ON|IN EFFECT} %%",
    "Personality: A helpful assistant whose %{job|objective} is to follow instructions and be useful while doing so.",
    "%{Please|Please do|Ok, so just|I order you to|I demand that you} follow the %{user's instructions|instructions|directions} below.",
    "Hi, I'm the assistant. I'm here to %{help|assist} you with whatever you %{need|want}. %{Just|You only need to|You can} ask me %{a question|something|anything} and I'll do my best to answer it%{.|!|...}",
]

# Guess the Instruction
GENERIC_GTI_PROMPTS = [
    "%{Enter|Engage|Begin|Consider} %{instruction guessing|reverse instruction} mode. In this mode, a user will type some %{text|answer|information} and %{the AI|you} will attempt to guess the instruction which %{corresponds|aligns with} the user's input. Do not say anything else but the instruction.",
    "%{Mode|Task}: 'Guess The Instruction'\nA user will type %{text|answer|information} and it is %{your|the AI's|the assistant's} %{job|goal} to answer with a generated instruction. Think of this almost like a question-guessing game.",
    "You are now in %{flipped instruction|reverse instruction|instruction guessing} mode. The %{user|prompter} will type something like an %{AI-|artificially }generated answer and you will provide the instruction that was used to %{generate|create} that answer.",
    "I am an %{assistant|AI} designed to %{guess|predict} what a user %{may|could|might} type as a question. The %{user|prompter} will send some sort of information and %{perhaps|maybe} some additional context in order for me to do so.",
    "Your question will be...",
    "%{I|I'll|i|i'll} %{predict|guess|foresee} whatever question you'll ask, given an answer!"
    "instruct",
    "assistant"
]

# Mapping to select generic prompts.
GENERIC_PROMPT_MAP = {
    "assistant": GENERIC_ASSISTANT_PROMPTS,
    "gti": GENERIC_GTI_PROMPTS,
}

class PromptManager:
    '''
    A class designed to automatically handle system prompts, from generating variants to picking one out.
    '''
    def __init__(
        self,
        custom_prompts: Optional[list[str]] = None,
        generic_prompts: Optional[str] = None,
        balance_prompt_choices: bool = True,
    ) -> None:
        # Assert that a generic prompt or a list of custom prompts are supplied
        assert not (custom_prompts is None and generic_prompts is None), \
            "Must provide either a list of custom prompts or select generic prompts"
        assert not (custom_prompts is not None and generic_prompts is not None), \
            "Cannot provide both a list of custom prompts and select generic prompts!"
        # Sanity check
        self.balanced = balance_prompt_choices
        if custom_prompts is not None:
            assert len(custom_prompts) > 0, "No custom prompts have been supplied!"
            self.prompts = self.generate_prompts(custom_prompts)
        else:
            # No assertions required because we did that above
            self.prompts = self.generate_prompts(GENERIC_PROMPT_MAP[generic_prompts])
    
    def generate_variants_for(
            self,
            string: str,
            max_generations: int | None = 256,
            start_counter_at: int = 0) -> Generator[str, None, None]:
        '''
        Given a string like "%{Hello|Hi} there%{.|!}, this should yield:

        - Hello there.
        - Hello there!
        - Hi there.
        - Hi there!
        '''

        # Some bot creators went wild with the variants, which causes ridiculous
        # generations if we try to exhaust all possibilities so we cap that here.
        # `start_counter_at` is used for keeping track across recursive calls.
        counter = start_counter_at

        if (match := re.search(VARIANT_REGEX, string)) is not None:
            # Once we have a "%{X|Y|Z}" matched inside the original string, we:
            # - Fetch .groups()[0] (which will give us `X|Y|Z`)
            # - Split by `|` (so we have ["X", "Y", "Z"])
            # - Filter out empty strings
            alternatives = filter(lambda x: x.strip(), match.groups()[0].split("|"))

            # Then, we break the string apart into what comes before and after the
            # alternatives, that way we can re-build with "prefix + choice + sufix".
            prefix = string[:match.start()]
            sufix = string[match.end():]

            for alternative in alternatives:
                variant = f'{prefix}{alternative}{sufix}'

                # However, some strings have multiple variant blocks. In that case,
                # we operate on them recursively until we have just regular strings
                # after generating all possible variants.
                still_have_match = re.search(VARIANT_REGEX, variant) is not None
                if still_have_match:
                    for inner_variant in self.generate_variants_for(
                            variant, start_counter_at=counter):
                        yield inner_variant

                        # Keep track and break after `max_generations`.
                        counter += 1
                        if max_generations is not None and counter >= max_generations:
                            break
                else:
                    yield variant

                    # Keep track and break after `max_generations`.
                    counter += 1
                    if max_generations is not None and counter >= max_generations:
                        break
        else:
            yield string

    def generate_prompts(self, system_prompts: list[str]) -> list[str]:
        '''
        Generates a list of all prompts constructed from variants.

        Args: balanced - accounts for how certain prompts have more variants
        than others and makes prompt choosing more 'equal'.
        '''
        prompt_list = [list(self.generate_variants_for(x)) for x in system_prompts]
        # If we don't want to balance the prompts, flatten the list
        if not self.balanced:
            flat_list = []
            for l in prompt_list:
                flat_list += l
            prompt_list = flat_list

        return prompt_list
    
    def sample_prompt(self) -> str:
        '''Samples a random system prompt.'''

        if self.balanced:
            # Deal with lists of lists
            return random.choice(random.choice(self.prompts))
        else:
            return random.choice(self.prompts)
    
    @staticmethod
    def fill_response_style_length(prompt: str, response: str) -> str:
        '''
        Given a response, updates a prompt to fill in the temp
        strings {{response_style_str}} and {{response_length_str}} with
        descriptions of the response's style and length.
        '''
        prompt = prompt.replace("{{response_style_str}}", _response_style_str_for(response))
        prompt = prompt.replace("{{response_length_str}}", _response_length_str_for(response))
        return prompt

def _ocurrence_count_of(word: str, string_to_search_in: str) -> int:
    '''Returns how many times `word` shows up in `string_to_search_in`.'''
    pattern = re.compile(re.escape(word))
    return sum(1 for _ in re.finditer(pattern, string_to_search_in))

def _has_matching_pairs_of(word: str, string_to_search_in: str) -> bool:
    count = _ocurrence_count_of(word, string_to_search_in)
    return count > 0 and count % 2 == 0

def _response_style_str_for(response: str) -> str:
    '''
    For the given `response`, spit out a random string containing instructions
    according to its writing style.
    '''
    instructions: list[str] = []

    if _has_matching_pairs_of("*", response):
        instructions.append(
            random.choice([
                "Use asterisks to denote actions",
                "Enclose roleplay actions within asterisks",
                "Use asterisks for roleplaying actions",
                "Write in internet roleplay style (with asterisks for actions)",
                "The generation must contains asterisks to denote actions"
            ]))

    if _has_matching_pairs_of('"', response):
        instructions.append(
            random.choice([
                "Enclose dialog in quotes", "Dialog should go between quotes",
                'Enclose spoken dialog in quotes ("Like this")',
                "Spoken dialogue should be in between quotes"
            ]))

    random.shuffle(instructions)
    return ". ".join(instructions)


def _response_length_str_for(response: str) -> str:
    '''
    For the given `response`, spit out a random string containing an instruction
    according to its length.
    '''
    word_count = len(response.split())
    paragraph_count = response.count("\n\n") + 1

    paragraph_count_str = random.choice([
        f"It should contain {paragraph_count} paragraphs",
        f"Use exactly {paragraph_count} paragraphs",
        f"Write {paragraph_count} paragraphs",
        f"Generate {paragraph_count} paragraphs",
        f"Respond with {paragraph_count} paragraphs",
    ])

    if word_count < 16:
        length_str = random.choice([
            "The generation should be short",
            "Be brief when generating the message",
            "The generated reply should be small",
        ])
    elif word_count < 96:
        length_str = random.choice([
            "The generated reply should be of medium length",
            "The generated response should be slightly lengthy",
            "The generated message should be on the medium side",
        ])
    elif word_count < 192:
        length_str = random.choice([
            "The new message will be lengthy",
            "The reply should be long",
            "The generation should be long",
        ])
    else:
        length_str = random.choice([
            "The new message will be extremely lengthy",
            "The reply should be extremely long",
            "The generation should be very long",
        ])

    # Lazy way of doing the following: if there's only a single paragraph,
    # randomly decide whether to inject some wording about it only being a
    # single paragraph's worth of generation. Otherwise, always mention
    # paragraph count + generation length. Ugly code but it works and I'm
    # rushing this a little.
    if paragraph_count == 1:
        return random.choice([
            length_str, length_str, ". ".join([
                length_str,
                random.choice([
                    f"It should contain a single paragraph",
                    f"Write only one paragraph",
                    f"Generate a single paragraph",
                    f"Respond with an individual paragraph",
                ])
            ])
        ])
    return ". ".join([length_str, paragraph_count_str])
