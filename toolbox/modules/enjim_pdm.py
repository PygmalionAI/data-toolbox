import logging
import typing as t
from functools import lru_cache

import yaml
from transformers import pipeline

from toolbox.core.models import Episode, Turn
from toolbox.datasets.enjim import EnjimDataset, EnjimAgent
from toolbox.modules import BaseModule
from toolbox.modules.registry import ModuleRegistry


class EnjimPDM(BaseModule, metaclass=ModuleRegistry):
    """Persona Dialogue Module based on the Enjim dataset."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        with open('resources/config.yaml', 'r') as configuration_file:
            self.settings = yaml.safe_load(configuration_file)['enjim']
        self.summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")

    def generator(self) -> t.Generator[Episode, None, None]:
        for episode in EnjimDataset():
            thread_subject: str = episode.thread_subject
            agents: t.Dict[str, EnjimAgent] = episode.agents
            posts: t.List[t.Tuple[str, str]] = episode.posts
            bot_name = posts[0][0]
            turns = [Turn(utterance=spoken, speaker=speaker, human_speaker=speaker != bot_name) for speaker, spoken in
                     posts]
            participant_personas = {ag.name: self.summarize_char(ag)
                                    if len(ag.persona) > self.settings['max_persona_chars']
                                    else ag.persona for ag in agents.values()}
            # Scenario is just the summary of the first post.
            world_scenario = thread_subject + ": " + posts[0][1]
            if len(world_scenario) > self.settings['max_scenario_chars']:
                world_scenario = self.summarize(world_scenario)
            yield Episode(turns=turns, participant_personas=participant_personas, world_scenario=world_scenario)

    @lru_cache(10_000)
    def summarize_char(self, character: EnjimAgent):
        return self.summarize(character.persona)

    def summarize(self, text: str):
        if len(text) > self.settings['summary_char_limit']:
            for separator in ['. ', '\n']:
                if separator in text:
                    sents = text.split(separator)
                    halfway = int(len(sents) / 2)
                    return self.summarize(separator.join(sents[:halfway])) + ' ' + self.summarize(
                        ". ".join(sents[halfway:]))
        return self.summarizer(text)[0]['summary_text']
