import logging
import typing as t

import tokenizers

from toolbox.core.consts import PromptConstants
from toolbox.core.models import Episode, SupervisedExample, Turn

LOG = logging.getLogger(__name__)

# NOTE: When processing episodes down into training examples, tokenizing text to
# get an accurate token count is a massive bottleneck (~49.5% of CPU time). We
# can instead use an estimation instead if we're OK with dropping some examples
# at training time.
AVG_TOKEN_TO_WORD_RATIO = 1.7
RETURN_ESTIMATION = True


class SupervisedExampleGenerator:
    '''
    Processes an episode down to supervised training examples.

    This involves parsing given episodes down into text, carefully balancing the
    amount of turns so everything can fit within the configured model's context
    window after being tokenized.
    '''

    def __init__(self, tokenizer_name: str, target_length: int) -> None:
        '''
        :param tokenizer_name: The name of the tokenizer to feed to
            HuggingFace's `from_pretrained()`
        :param target_length: Usually, the model's maximum context size.
        '''
        self.tokenizer: tokenizers.Tokenizer = tokenizers.Tokenizer.from_pretrained(
            tokenizer_name)
        self.target_length = target_length

        super().__init__()

    def process(
        self, episode: Episode
    ) -> t.Generator[t.Tuple[Episode, SupervisedExample], None, None]:
        # Start off with the persona data at the top.
        base_prompt = ""
        for speaker, persona in episode.participant_personas.items():
            pdm_prefix = PromptConstants.pdm_prefix_for(speaker)
            base_prompt += f"{pdm_prefix}: {persona}\n"

        # Then, world scenario.
        if episode.world_scenario:
            scenario_str = f"Scenario: {episode.world_scenario}\n"
            base_prompt += scenario_str

        base_prompt += f"{PromptConstants.CHAT_START_TOKEN}\n"
        base_len = self._tokenized_length(base_prompt)

        if base_len > self.target_length:
            LOG.warning(
                "Episode goes over context length without even adding turns (%s > %s): `%s`, skipping...",
                base_len, self.target_length, episode)
            return

        # Afterwards, we start dealing with chat history that can be broken
        # apart into separate training examples.
        cur_len = base_len
        cur_turns: list[Turn] = []

        for turn in episode.turns:
            if cur_len + self._turn_length(turn) > self.target_length:
                # Can't add this turn into this context window. Take what we
                # already have and yield it, then add this turn to the next
                # window.
                cur_len = base_len + self._turn_length(turn)
                cur_turns = [turn]
            else:
                # Turn fits! Add to current context window.
                cur_turns.append(turn)
                cur_len += self._turn_length(turn)

            output = self._example_and_episode_from_data(
                episode, cur_turns, base_prompt)
            if output is None:
                continue
            yield output

    def _example_and_episode_from_data(
            self, original_episode: Episode, cur_turns: list[Turn],
            base_prompt: str) -> None | t.Tuple[Episode, SupervisedExample]:
        if len(cur_turns) < 2:
            return None

        # If we have enough turns and the last one is not from a human, we
        # can yield a training example.
        last_turn = cur_turns[-1]
        if last_turn and not last_turn.human_speaker:
            # Collapse `cur_turns` down into text.
            prompt = base_prompt
            prompt += "\n".join(
                [f"{t.speaker}: {t.utterance.strip()}" for t in cur_turns[:-1]])

            # Append response prefix into `cur_prompt`, and yield the
            # example.
            prompt += f"\n{last_turn.speaker}:"
            trimmed_episode = Episode(
                turns=cur_turns,
                participant_personas=original_episode.participant_personas,
                world_scenario=original_episode.world_scenario,
            )
            example = SupervisedExample(prompt=prompt,
                                        response=last_turn.utterance.strip())

            # Sanity check so I can catch this easier in case I break
            # something.
            example_len = self._tokenized_length(prompt +
                                                 last_turn.utterance.strip())
            if example_len > self.target_length:
                LOG.warning(
                    f"Generated an example too large ({example_len} > {self.target_length})"
                )

            # NOTE: Needing to yield both the trimmed episode _and_ the
            # formatted training example is a side-effect of bad modeling on
            # my end. The filters expect episodes, but if we apply them on
            # the non-trimmed episodes we lose out on way too much data, so
            # we yield the trimmed version so we can run _those_ through the
            # filters instead.
            return trimmed_episode, example
        return None

    def _turn_length(self, turn: Turn) -> int:
        '''Returns the length of the given `turn`, in tokens.'''
        return self._tokenized_length(f"{turn.speaker}: {turn.utterance}")

    def _tokenized_length(self, string: str) -> int:
        '''Returns the length of the given `string`, in tokens.'''
        if RETURN_ESTIMATION:
            word_count = len(string.split())
            return round(word_count * AVG_TOKEN_TO_WORD_RATIO)
        return len(self.tokenizer.encode(string))
