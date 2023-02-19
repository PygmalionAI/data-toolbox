import logging
from functools import lru_cache
from re import search
from typing import Optional, Generator, Tuple, List, Dict

from transformers import pipeline

from toolbox.core.models import Episode, Turn
from toolbox.datasets.enjim import EnjimDataset, EnjimAgent, setup_sqlite
from toolbox.modules import BaseModule
from toolbox.modules.registry import ModuleRegistry
from toolbox.parsers.bb_code import BBCtoMD
from toolbox.utils.dataset import get_data_path, get_config


class EnjimPDM(BaseModule, metaclass=ModuleRegistry):
    """
    Persona Dialogue Module based on the Enjim dataset.
    NOTE: All the summarizing stuff is just there until whatever is going to be done with vector dbs is figured out.
    """
    CACHE_QUERY = "SELECT summary FROM summary_cache WHERE forum_shortname = ? AND text_id = ? AND max_length = ?"
    CACHE_INSERT = "INSERT INTO summary_cache (forum_shortname, text_id, max_length, summary) VALUES (?, ?, ?, ?)"

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.settings = get_config('enjim')
        self.summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
        self.cache_db = setup_sqlite(get_data_path("enjim"), EnjimDataset.CACHE_DB, self.settings['cache_db'],
                                     logger=self.logger)

    def generator(self) -> Generator[Episode, None, None]:
        for episode in EnjimDataset():
            thread_subject: str = episode.thread_subject
            agents: Dict[str, EnjimAgent] = episode.agents
            posts: List[Tuple[str, str]] = episode.posts
            bot_name = posts[0][0]
            turns = [Turn(utterance=spoken if not search(BBCtoMD.INVALID_RESULT, spoken)
                     else self.summarize(spoken, self.settings['max_scenario_chars'], None, None),
                          speaker=speaker, human_speaker=speaker != bot_name) for speaker, spoken in
                     posts]
            participant_personas = {ag.name: self.summarize_char(ag, self.settings['max_persona_chars'],
                                                                 episode.forum_shortname) for ag in agents.values()}
            # Scenario is just the summary of the first post.
            world_scenario = thread_subject + ": " + posts[0][1]
            world_scenario = self.summarize(world_scenario, self.settings['max_scenario_chars'],
                                            episode.thread_id, episode.forum_shortname)
            yield Episode(turns=turns, participant_personas=participant_personas, world_scenario=world_scenario)

    @lru_cache(10_000)
    def summarize_char(self, character: EnjimAgent, max_length: int, forum_shortname):
        if len(character.persona) <= max_length and not search(BBCtoMD.INVALID_RESULT, character.persona):
            return character.persona
        return self.summarize(character.persona, max_length, character.name+character.user_id, forum_shortname)

    def summarize(self, text: str, max_length: int, thread_or_user_id: Optional[str], forum_shortname: Optional[str]):
        if forum_shortname is not None:
            for summary in self.cache_db.execute(self.CACHE_QUERY, (forum_shortname, thread_or_user_id, max_length)):
                return summary
        combined_summary = None
        if len(text) > self.settings['summary_char_limit']:
            for separator in ['. ', '\n']:
                if separator in text:
                    sents = text.split(separator)
                    halfway = int(len(sents) / 2)
                    first = self.summarize(separator.join(sents[:halfway]), max_length, None, None)
                    second = self.summarize(separator.join(sents[halfway:]), max_length, None, None)
                    combined_summary = first + separator + second
                    while len(combined_summary) > max_length:
                        combined_summary = self.summarize(combined_summary, max_length, None, None)
                    break
        else:
            combined_summary = self.summarizer(text)[0]['summary_text']
        if forum_shortname is not None:
            self.cache_db.execute(self.CACHE_INSERT, (forum_shortname, thread_or_user_id, max_length, combined_summary))
            self.cache_db.commit()
        return combined_summary
