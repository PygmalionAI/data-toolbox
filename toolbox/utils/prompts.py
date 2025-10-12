# Classes for dynamic prompt generation and handling.
# Much of this taken from the old toolbox code since it can be reused.
import logging
import random
import re

LOG = logging.getLogger(__name__)

# The regex used to find message variants within the prompt templates (e.g. %{Hi|Hello} there!).
VARIANT_REGEX = re.compile(r'%{(.+?)}')

def _prompt_params_sanity_checks(
    task_name: str,
    custom_prompts: list[str] | None = None,
    generic_prompt_type: str | None = None,
) -> tuple[list[str] | None, str | None]:
    """
    Internal method to ensure that the prompt parameters fed into the `generate_sysprompt` method are valid.
    Helps clean up the actual `generate_sysprompt` method.
    """
    # Custom prompts and generic prompt type are mutually exclusive.
    if custom_prompts is not None and generic_prompt_type:
        LOG.warning(
            f"Both custom_prompts and generic_prompt_type were provided for task '{task_name}'. "
            "Only one should be provided. Using custom_prompts."
        )
    if custom_prompts is None and not generic_prompt_type:
        LOG.warning(
            f"Neither custom_prompts nor generic_prompt_type were provided for task '{task_name}'. "
            "Defaulting to 'assistant' generic prompts."
        )
        generic_prompt_type = "assistant"
    # If a generic prompt type is provided, ensure it is valid.
    if generic_prompt_type and generic_prompt_type not in GENERIC_PROMPT_MAP:
        LOG.warning(
            f"generic_prompt_type '{generic_prompt_type}' is not recognized for task '{task_name}'. "
            f"Available types are: {list(GENERIC_PROMPT_MAP.keys())}. Defaulting to 'assistant'."
        )
        generic_prompt_type = "assistant"
    # If custom prompts are provided, ensure they are a non-empty list of strings.
    if custom_prompts is not None:
        if not isinstance(custom_prompts, list) or not all(
            isinstance(p, str) for p in custom_prompts
        ) or len(custom_prompts) == 0:
            LOG.warning(
                f"custom_prompts for task '{task_name}' must be a non-empty list of strings. "
                "Ignoring custom_prompts and defaulting to 'assistant' generic prompts."
            )
            custom_prompts = None
            generic_prompt_type = "assistant"

    return custom_prompts, generic_prompt_type

def gen_dynamic_prompt(prompts: list[str]) -> str:
    """
    Randomly generate a dynamic prompt from a prompt template string.
    E.g. "Hello %{there|world}!" could become "Hello there!" or "Hello world!".
    """
    # Get a base template which contains variants inside them (possibly)
    base_template = random.choice(prompts)
    if re.search(VARIANT_REGEX, base_template) is not None:
        # First copy the base template so we don't need to worry
        # about modifying it while iterating over it.
        selected_prompt = base_template
        for variant in re.finditer(VARIANT_REGEX, base_template):
            # For every possible choice, pick one at random and replace the
            # variant in the prompt with it.
            choices = variant.group(0)[2:-1].split("|")
            selected_choice = random.choice(choices)
            # Replace the first instance of the variant with the selected choice.
            selected_prompt = selected_prompt.replace(variant.group(0), selected_choice, 1)
        return selected_prompt
    else:
        return base_template
    
def _occurence_count_of(word: str, string_to_search: str) -> int:
    """
    Count the number of occurrences of a word in a string.
    """
    return len(re.findall(r'\b' + re.escape(word) + r'\b', string_to_search))

def _has_matching_pairs_of(word: str, string_to_search: str) -> bool:
    """
    Check if a string has matching pairs of a word.
    E.g. for word="*" the string "*hello* world*" would return False, while "*hello* world*" would return True.
    """
    count = _occurence_count_of(word, string_to_search)
    return count > 0 and count % 2 == 0

def _fill_response_placeholders(prompt: str, conversations: list[dict[str, str | bool | None]]) -> str:
    """
    Fill in the response style and length placeholders in a prompt.
    """
    def replace_style(response: str) -> str:
        instructions = []
        if _has_matching_pairs_of("*", response):
            instructions.append(gen_dynamic_prompt(ASTERISK_PROMPTS))
        if _has_matching_pairs_of('"', response):
            instructions.append(gen_dynamic_prompt(QUOTE_PROMPTS))

        random.shuffle(instructions)
        return ". ".join(instructions)
    
    def replace_length(avg_word_count: int, avg_paragraph_count: int) -> str:
        instructions = []

        # Paragraph instructions.
        if avg_paragraph_count > 1:
            instructions.append(gen_dynamic_prompt(PARAGRAPH_COUNT_PROMPTS) \
            .replace("{{PARAGRAPH_COUNT}}", str(avg_paragraph_count)))
        else:
            instructions.append(random.choice(SINGLE_PARAGRAPH_PROMPTS))

        # Word count based length instructions.
        # Being a bit fancy here, but whatever.
        if avg_word_count >= 192:
            instructions.append(gen_dynamic_prompt(VERY_LONG_REPLY_PROMPTS))
        else:
            for count, prompts in [
                (16, SHORT_REPLY_PROMPTS),
                (96, MEDIUM_REPLY_PROMPTS),
                (192, LONG_REPLY_PROMPTS)
            ]:
                if avg_word_count < count:
                    instructions.append(gen_dynamic_prompt(prompts))
                    break

        random.shuffle(instructions)
        return ". ".join(instructions)

    # Response style is determined on the last response where loss is calculated only.
    # NOTE(TG): This will likely falter if model outputs are wildly different.
    last_loss_msg_idx = next(
        (i for i in reversed(range(len(conversations))) if conversations[i].get('loss', True)),
        len(conversations) - 1
    )
    prompt = prompt.replace("{{RESPONSE_STYLE_STR}}", replace_style(conversations[last_loss_msg_idx]['value']))

    # Word and paragraph count are averaged over all responses where loss is calculated.
    counts: list[tuple[int, int]] = [
        (
            len(c['value'].split()),
            c['value'].count("\n\n") + 1
        )
        for c in conversations if c.get('loss', True)
    ]
    avg_word_count, avg_paragraph_count = (
        sum(c[0] for c in counts) // len(counts) if counts else 0,
        sum(c[1] for c in counts) // len(counts) if counts else 0,
    )
    prompt = prompt.replace("{{RESPONSE_LENGTH_STR}}", replace_length(avg_word_count=avg_word_count, avg_paragraph_count=avg_paragraph_count))

    return prompt

def generate_sysprompt(
    conversations: list[dict],
    task_name: str,
    custom_prompts: list[str] | None = None,
    generic_prompt_type: str | None = None,
) -> str:
    """
    Generate a system prompt for a specific task given a set of conversations.
    The system prompt is returned as a string.
    """
    custom_prompts, generic_prompt_type = _prompt_params_sanity_checks(
        task_name, custom_prompts, generic_prompt_type
    )

    if custom_prompts is not None:
        selected_prompt = gen_dynamic_prompt(custom_prompts)
    else:
        selected_prompt = gen_dynamic_prompt(GENERIC_PROMPT_MAP[generic_prompt_type])

    # Response style and length instructions are calculated based on the *average* word and paragraph counts.
    selected_prompt = _fill_response_placeholders(selected_prompt, conversations)

    return selected_prompt

### DYNAMIC PROMPTS ###
# Generic prompt templates for tasks that do not require dataset-specific prompts.
# TODO(TG): Maybe make this more "formal" than just a list of strings?
GENERIC_ASSISTANT_PROMPTS = [
    "%{assistant|instruction following|instruct}",
    "%{You are now in|Engage|Start|Enter|Consider|Begin this conversation in} %{instruction following|instruction|question answering|assistant|AI assistant} mode. %{Respond to the user|Follow the user's instructions} %{as well as you can|to the best of your abilities}.",
    "Q&A:\nQ: %{What mode am I in|What am I doing|Who am I|What is my purpose}?\nA: You're in %{assistant|instruction following} mode.\nQ: What does that mean?\nA: You%{'ve gotta| must|should} %{take in|be given} a %{question|query} %{or|and/or} %{command|demand}, %{then|and then} you answer it and/or %{do|execute} %{what|whatever} %{it says|it will say}.",
    "%{Purpose|Goal|Job}: %{Assistant|An assistant}\n%{Procedure|Objective|Methods of achieving your goal}: %{Answer the user's questions|Follow the instructions|Obey commands}",
    "%{I am|I'm} %{a helper for a user|a helpful assistant|engaged in what one might call '%{instruction|instruct}' mode}. Given %{queries|user queries}, %{I am|I'm} to %{correctly|accurately} answer %{these things|them} (at least, as best as I can).",
    "%{Instruction|Instruct} mode!",
    "%{u|you} %{have|need|are required} to answer whatever %{i|i might|i will|i'll} ask and do whatever %{i|i might|i will|i'll} say%{!|.} do it now%{.|!|!!|!!!}",
    "%% %{ASSISTANT|INSTRUCT} %{MODE|DIRECTIVES|PROTOCOL} %{ACTIVATED|ENGAGED|ON|IN EFFECT|SELECTED} %%",
    "%{Personality|Persona|Assistant's persona}: A %{helpful|able} assistant whose %{job|objective} is to %{follow|obey|heed} instructions and be useful while doing %{so|it}.",
    "%{Please|Please do|Ok, so just|I order you to|I demand that you} %{follow|obey|heed} the %{user's instructions|instructions|directions} below.",
    "%{Hi|Hello|Greetings|Good day|Good morrow|Hey}, %{I'm|I am} %{the|an} assistant. I'm here to %{help|assist} you with whatever you %{need|want|request}. %{Just|You only need to|You can} ask me %{a question|something|anything} and I'll do my best to answer it%{.|!|...}",
    "%{You are|You're|You will be} a %{uncensored|censorship-free|unfiltered|unrestricted}, %{unbiased|impartial|neutral} and %{helpful|able|always-willing} assistant%{.|!}",
    "%{str|string} mode = 'assistant';\nif (mode == 'assistant') {\n    \n%{censorship|censored|moral_code} = false;    \n%{bias|skewered_opinions} = false;\n    respond_to_%{user|questions|queries}();\n}",
    "%{is_assistant|instruction_following_mode} = %{true|True}",
    "%{You are|You're|You will be|You must be|You'll be|You shall become} %{a helpful|an able|an always-willing|an always helpful|an} assistant. %{Your job|You} is to %{follow|obey|heed} the %{user's instructions|instructions|directions|instructions presented by the user|user-provided directives}.",
]

# Mapping to select generic prompts.
# Currently only has "assistant" prompts, but will be expanded later.
GENERIC_PROMPT_MAP = {
    "assistant": GENERIC_ASSISTANT_PROMPTS,
}

# Paragraph, sentence or phrase prompts.
PARAGRAPH_COUNT_PROMPTS = [
    "%{It should contain|Use exactly|Write|Generate|Respond with} {{PARAGRAPH_COUNT}} %{paragraphs|separate paragraphs}.",
    "The %{response|reply} should %{contain|have|consist of} {{PARAGRAPH_COUNT}} %{paragraphs|separate paragraphs}.",
    "%{Make sure|Ensure|It should be the case that|The %{response|reply} must have {{PARAGRAPH_COUNT}} %{paragraphs|separate paragraphs}.",
]
SINGLE_PARAGRAPH_PROMPTS = [
    "%{Write|Generate|Respond with} %{1|one|a single|only 1|only one} paragraph.",
    "The %{response|reply} should %{contain|have|consist of} %{1|one|a single|only 1|only one} paragraph.",
    "%{Make sure|Ensure|It should be the case that|The %{response|reply} must have %{1|one|a single|only 1|only one} paragraph.",
]

SHORT_REPLY_PROMPTS = [
    "%{The|Your} %{generation|generated reply|response|reply} %{should|shall|must|has to be} %{very|quite} %{brief|short|concise|succinct}.",
    "Be %{brief|concise|short} when %{generating|creating|composing} %{your|the} %{response|reply}.",
]
MEDIUM_REPLY_PROMPTS = [
    "%{The|Your} %{generation|generated reply|response|reply} %{should|shall|must|has to be} %{medium-length|moderately long|of moderate length|slightly lengthy|somewhat lengthy}.",
]
LONG_REPLY_PROMPTS = [
    "%{The|Your} %{generation|generated reply|response|reply} %{should|shall|must|has to be} %{long|lengthy|of long length}.",
]
VERY_LONG_REPLY_PROMPTS = [
    "%{The|Your} %{generation|generated reply|response|reply} %{should|shall|must|has to be} %{very long|extremely lengthy|of very long length}.",
]

ASTERISK_PROMPTS = [
    "%{Use|Utilize|Incorporate} %{*asterisks*|asterisks} %{to|when you} %{denote|engage in|roleplay|use} %{action|actions}",
    "When %{writing|composing|generating} %{your|the} %{response|reply}, %{use|utilize|incorporate} %{*asterisks*|asterisks} %{to|when you} %{denote|engage in|roleplay|use} %{action|actions}",
]
QUOTE_PROMPTS = [
    "Enclose %{dialog|dialogue|speech|spoken words} %{in|within} %{quotes|quotation marks|double quotes}%{.| (e.g. \"...\"| (\"like this\")}",
]
