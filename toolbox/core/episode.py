from calendar import c
import logging
import typing as t

import tokenizers

from toolbox.core.models import Episode, SupervisedExample, Turn
from toolbox.core.consts import PromptConstants

LOG = logging.getLogger(__name__)


class SupervisedEpisodeProcessor:
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
        self.tokenizer = tokenizers.Tokenizer.from_pretrained(tokenizer_name)
        self.target_length = target_length

        super().__init__()

    def process(self,
                episode: Episode) -> t.Generator[SupervisedExample, None, None]:
        # Start off with the persona data at the top.
        base_prompt = ""
        for speaker, persona in episode.participant_personas.items():
            base_prompt += f"{speaker}'s Persona: {persona}\n"

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
        cur_prompt = base_prompt
        cur_len = base_len
        cur_turns: list[Turn] = []

        for turn in episode.turns:
            last_turn = cur_turns[-1]

            # If we have enough turns and the last one is not from a human, we
            # can yield a training example.
            if len(cur_turns) > 1 and not last_turn.human_speaker:
                # Collapse `cur_turns` down into text and append to `cur_prompt`
                cur_prompt += "\n".join(
                    [f"{t.speaker}: {t.utterance}" for t in cur_turns[:-1]])

                # Append response prefix into `cur_prompt`, and yield the
                # example.
                cur_prompt += f"{last_turn.speaker}:"
                yield SupervisedExample(prompt=cur_prompt,
                                        response=last_turn.utterance)

            if cur_len + self._turn_length(turn) > self.target_length:
                # Can't add this turn into this context window. Take what we
                # already have and yield it, then add this turn to the next
                # window.
                cur_prompt = base_prompt
                cur_len = base_len + self._turn_length(turn)
                cur_turns = [turn]
            else:
                # Turn fits! Add to current context window.
                cur_turns.append(turn)
                cur_len += self._turn_length(turn)

    def _turn_length(self, turn: Turn) -> int:
        '''Returns the length of the given `turn`, in tokens.'''
        return self._tokenized_length(f"{turn.speaker}: {turn.utterance}")

    def _tokenized_length(self, string: str) -> int:
        '''Returns the length of the given `string`, in tokens.'''
        return len(self.tokenizer(string).input_ids)
