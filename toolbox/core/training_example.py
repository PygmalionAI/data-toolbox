import logging
import math
import random
import re
import typing as t

from toolbox.core.models import (
    Episode,
    TrainingExample,
    TurnKind
)

LOG = logging.getLogger(__name__)

# NOTE: When processing episodes down into training examples, tokenizing text to
# get an accurate token count is a massive bottleneck (~49.5% of CPU time). We
# can instead use an estimation instead if we're OK with dropping some examples
# at training time.
AVG_WORD_TO_TOKEN_RATIO = 1.7

class TurnTooLargeError(RuntimeError):
    pass

class TrainingExampleGenerator:
    '''Converts an `Episode` into `TrainingExample`s.'''

    def __init__(
        self,
        episode: Episode,
        target_token_count: int = 2048,
        format: str = "metharme"
    ) -> None:
        self.episode = episode
        self.format = format.lower()
        assert self.format in ["pygmalion", "metharme"], "Invalid format specified!"

        # Minus 32 is to account for the special tokens that we replace in the
        # input prompt, which will likely cause the prompt to expand.
        self.target_token_count = target_token_count - 32

        # Different formats have different functions for converting
        # a Turn into a string. Do an if statement here so we don't have
        # to do a bunch more later.
        if self.format == "metharme":
            self.turn_to_str = lambda x: x.as_meth_str()
        else:
            self.turn_to_str = lambda x: x.as_pyg_str()

        super().__init__()

    def __iter__(self) -> t.Generator[TrainingExample, None, None]:
        examples_yielded = 0

        # Always start off with the system turn.
        system_turn = self.episode.turns[0]
        assert system_turn.kind == TurnKind.SYSTEM
        base_turns = [system_turn]

        cur_turns = base_turns.copy()
        cur_len = _token_count_for(self.turn_to_str(system_turn))

        for turn in self.episode.turns[1:]:
            turn_len = _token_count_for(self.turn_to_str(turn))

            if cur_len + turn_len > self.target_token_count:
                # Can't add this turn into the context window. Start dropping
                # older turns into we can fit it in here.
                len_over_target = math.inf

                while len_over_target > 0:
                    try:
                        removed_turn = cur_turns.pop(1)
                        cur_len -= _token_count_for(self.turn_to_str(removed_turn))

                        len_over_target = self.target_token_count - (cur_len +
                                                                     turn_len)
                    except IndexError as ex:
                        raise TurnTooLargeError from ex

            # We have space for the next turn, so add it to the context window.
            cur_turns.append(turn)
            cur_len += _token_count_for(self.turn_to_str(turn))

            # Yield training example if this is a model turn.
            if turn.kind != TurnKind.MODEL:
                continue

            # The prompt is comprised of every single turn converted into its
            # string representation, _except_ for the last model turn. For the
            # last model turn, we append the TurnKind.MODEL token to the end of
            # the prompt, and then use the model's utterance as the response.
            prompt = "".join([self.turn_to_str(t) for t in cur_turns[:-1]])
            if self.format == "metharme":
                prompt += TurnKind.MODEL.value
            else:
                prompt += f"\n{turn.name}: "
            generation = turn.utterance.strip()

            # Sanity checks. Asserts that there's only a single system prompt
            # and it's at the very beginning of the prompt string.
            try:
                # NOTE(11b): Some datasets now include multiple system prompts
                # so I'm turning off this check for now. Reconsider later.
                # assert _ocurrence_count_of(TurnKind.SYSTEM.value, prompt) == 1
                if self.format == "metharme":
                    assert prompt.find(TurnKind.SYSTEM.value) == 0
            except AssertionError as ex:
                LOG.error(
                    "Sanity checks for generated training example failed.")
                LOG.error("Prompt: %s", prompt)
                LOG.error("Generation: %s", generation)
                raise ex

            # TODO(11b): This is probably not the greatest place for this, but
            # would require a decent amount of rework to put at the task level
            # depending on the task so let's roll with this for now.
            prompt = prompt.replace("{{response_style_str}}",
                                    _response_style_str_for(generation))
            prompt = prompt.replace("{{response_length_str}}",
                                    _response_length_str_for(generation))

            yield TrainingExample(
                prompt=prompt,
                generation=generation,
                identifier=f"{self.episode.identifier}-{examples_yielded}",
            )
            examples_yielded += 1


def _ocurrence_count_of(word: str, string_to_search_in: str) -> int:
    '''Returns how many times `word` shows up in `string_to_search_in`.'''
    pattern = re.compile(re.escape(word))
    return sum(1 for _ in re.finditer(pattern, string_to_search_in))


def _has_matching_pairs_of(word: str, string_to_search_in: str) -> bool:
    count = _ocurrence_count_of(word, string_to_search_in)
    return count > 0 and count % 2 == 0


def _token_count_for(string: str) -> int:
    return math.ceil(len(string.split()) * AVG_WORD_TO_TOKEN_RATIO)


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
