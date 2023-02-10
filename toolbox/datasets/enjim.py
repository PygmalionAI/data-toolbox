import logging
import sqlite3
import typing as t
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from os.path import isfile

import mashumaro
import requests
import yaml
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

from toolbox.datasets import BaseDataset
from toolbox.parsers.bb_code import BBCtoMD
from toolbox.utils.chunking import right_size
from toolbox.utils.dataset import get_data_path


@dataclass(frozen=True)
class EnjimAgent(mashumaro.DataClassDictMixin):
    name: str
    user_name: str
    user_id: str
    persona: str


@dataclass(frozen=True)
class EnjimEpisode(mashumaro.DataClassDictMixin):
    thread_subject: str
    agents: t.Dict[str, EnjimAgent]
    posts: t.List[t.Tuple[str, str]]


class EnjimDataset(BaseDataset[EnjimEpisode]):

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        with open('resources/config.yaml', 'r') as configuration_file:
            self.settings = yaml.safe_load(configuration_file)['enjim']
        self.root_data_path = get_data_path("enjim")
        self.parser = BBCtoMD(self.settings['img_recognition_model'])
        self.conns = None
        tokenizer = AutoTokenizer.from_pretrained(self.settings['ner_model'])
        model = AutoModelForTokenClassification.from_pretrained(self.settings['ner_model'])
        self.nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    def generator(self) -> t.Generator[EnjimEpisode, None, None]:
        self.load_sqlite()
        for shortname, configuration in self.settings['sources'].items():
            roleplay_forums = configuration['roleplay_forums']
            thread_query = f'SELECT thread_id, thread_subject FROM forum_threads WHERE forum_id ' \
                           f'IN ({", ".join(roleplay_forums)}) ORDER BY thread_views DESC'
            for idx, (rp_thread_id, rp_thread_subject) in enumerate(self.conns[shortname].execute(thread_query)):
                self.logger.info('Processing thread number %s: %s.', idx, rp_thread_subject)
                posts_query = f"SELECT post_content, post_user_id, post_username from forum_posts where thread_id = {rp_thread_id}"
                thread = [post for post in self.conns[shortname].execute(posts_query)]
                # Not doing at the filter stage for performance reasons
                if len(thread) < self.settings['min_posts_cutoff']:
                    self.logger.warning('Too short of a thread; only %s posts. Skipping.', len(thread))
                    continue
                non_op_pct = sum([1 for post in thread if post[1] != thread[0][1]]) / len(thread)
                if non_op_pct < self.settings['minimum_external_participation']:
                    self.logger.warning('Not enough non-OP posters in thread. Skipping.')
                    continue
                episode: EnjimEpisode = self.parse_thread(thread, shortname, rp_thread_subject)
                yield episode

    def parse_thread(self, thread, shortname, rp_thread_subject) -> EnjimEpisode:
        # todo make this coherent for others to read
        user_to_agent = {}
        agents: t.Dict[str, EnjimAgent] = {}
        posts: t.List[t.Tuple[str, str]] = []
        for post_content, post_user_id, post_username in thread:
            self.logger.debug("Post: %s.", post_username)
            formatted_post = self.parser.to_markdown(post_content)
            if post_username not in user_to_agent:
                entities = self.nlp(formatted_post)
                counts = defaultdict(int)
                for entity in entities:
                    if entity['entity_group'] == 'PER':
                        counts[entity['word']] += 1
                speaker = None
                for person, count in reversed(sorted(counts.items(), key=lambda item: item[1])):
                    speaker: t.Optional[EnjimAgent] = self.get_chardef(post_user_id, person, shortname)
                    if speaker is not None:
                        break
                if speaker is None:
                    self.logger.warning(
                        'No character found for user_id %s, post_username %s, forum %s. Will not have a persona.',
                        post_user_id, post_username, shortname)
                    speaker = EnjimAgent(name=post_username, user_name=post_username, user_id=post_user_id,
                                         persona='')
                agents[speaker.name] = speaker
                user_to_agent[post_username] = speaker.name
            speaker_name = user_to_agent[post_username]
            if len(formatted_post) > self.settings['max_utterance_chars']:
                sub_utterances = right_size(scenes=[formatted_post],
                                            max_length=self.settings['max_utterance_chars'])
                for sub_utterance in sub_utterances:
                    posts.append((speaker_name, sub_utterance))
            else:
                posts.append((speaker_name, formatted_post))
        return EnjimEpisode(thread_subject=rp_thread_subject, posts=posts, agents=agents)

    @lru_cache(maxsize=10_000)
    def get_chardef(self, user_id, char_name, shortname) -> t.Optional[EnjimAgent]:
        self.load_sqlite()
        self.logger.info('Making/fetching character named %s with user id %s for forum %s.', user_id, char_name,
                         shortname)
        char_forums = self.settings['sources'][shortname]['character_forums']
        query = f"SELECT t.thread_id, t.thread_subject, p.post_content, p.post_user_id, p.post_username " \
                f"FROM forum_threads t INNER JOIN forum_posts p ON p.thread_id = t.thread_id WHERE " \
                f"forum_id IN ({', '.join(char_forums)}) AND post_user_id = {user_id} ORDER BY p.thread_id, post_time ASC"
        for thread_id, thread_subject, post_content, post_user_id, post_username in self.conns[shortname].execute(
                query):
            if char_name in post_content or char_name in thread_subject:
                self.logger.info('Found character thread for %s.', char_name)
                persona = self.parser.to_markdown(post_content)
                character: EnjimAgent = EnjimAgent(name=thread_subject, user_name=post_username, user_id=user_id,
                                                   persona=persona)
                return character
            self.logger.warning('Thread %s not a match for character %s.', thread_subject, char_name)
        self.logger.debug('No character found for user_id %s, character name %s, forum %s. Will not have a persona.',
                          user_id, char_name, shortname)

    def load_sqlite(self, force_recreate=False):
        """
        Downloads the file(s) and sets up connection(s) to the database(s).
        :return:
        """
        if not force_recreate and self.conns is not None and len(self.conns) > 0:
            return
        self.conns = {}
        for shortname, configuration in self.settings['sources'].items():
            path = configuration['path']
            data_path = f"{self.root_data_path}/{shortname}.db"
            if not isfile(data_path):
                self.logger.info('Downloading dataset %s from %s.', shortname, path)
                dataset = requests.get(path)
                with open(data_path, 'wb') as f:
                    f.write(dataset.content)
            try:
                self.conns[shortname] = sqlite3.connect(data_path)
            except Exception as error:
                self.logger.error('Error while connecting with db %s, error: %s.', shortname, error)
                raise error
            finally:
                self.logger.info('Added db %s.', shortname)
