import logging
import sqlite3
import typing as t
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from os.path import isfile

import mashumaro
import requests
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

from toolbox.datasets import BaseDataset
from toolbox.parsers.bb_code import BBCtoMD
from toolbox.utils.chunking import right_size
from toolbox.utils.dataset import get_data_path, get_config


@dataclass(frozen=True)
class EnjimAgent(mashumaro.DataClassDictMixin):
    name: str
    user_name: str
    user_id: str
    persona: str


@dataclass(frozen=True)
class EnjimEpisode(mashumaro.DataClassDictMixin):
    forum_shortname: str
    thread_subject: str
    thread_id: str
    agents: t.Dict[str, EnjimAgent]
    posts: t.List[t.Tuple[str, str]]


class EnjimDataset(BaseDataset[EnjimEpisode]):
    CACHE_DB = 'cache'
    THREAD_SELECT = "SELECT t.thread_id, t.thread_subject, p.post_content, p.post_user_id, p.post_username " \
                    "FROM forum_threads t INNER JOIN forum_posts p ON p.thread_id = t.thread_id WHERE " \
                    "forum_id IN (?) AND post_user_id = ? ORDER BY p.thread_id, post_time"
    POSTS_QUERY = "SELECT post_content, post_user_id, post_username from forum_posts where thread_id = ?"

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.settings = get_config('enjim')
        self.root_data_path = get_data_path("enjim")
        self.conns = None
        self.load_sqlite()
        # noinspection PyUnresolvedReferences
        self.parser = BBCtoMD(self.settings['img_recognition_model'], self.conns[self.CACHE_DB])
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
                thread = [post for post in self.conns[shortname].execute(self.POSTS_QUERY, (rp_thread_id, ))]
                # Not doing at the filter stage for performance reasons
                if len(thread) < self.settings['min_posts_cutoff']:
                    self.logger.warning('Too short of a thread; only %s posts. Skipping.', len(thread))
                    continue
                non_op_pct = sum([1 for post in thread if post[1] != thread[0][1]]) / len(thread)
                if non_op_pct < self.settings['minimum_external_participation']:
                    self.logger.warning('Not enough non-OP posters in thread. Skipping.')
                    continue
                episode: t.Optional[EnjimEpisode] = self.parse_thread(thread, shortname, rp_thread_subject,
                                                                      rp_thread_id)
                if episode is not None:
                    yield episode

    def parse_thread(self, thread, shortname, rp_thread_subject, thread_id) -> t.Optional[EnjimEpisode]:
        user_to_agent = {}
        agents: t.Dict[str, EnjimAgent] = {}
        posts: t.List[t.Tuple[str, str]] = []
        for post_content, post_user_id, post_username in thread:
            self.logger.debug("Post: %s.", post_username)
            formatted_post = self.parser.to_markdown(post_content)
            if len(formatted_post) > self.settings['cutoff_utterance_chars']:
                self.logger.warning('Too long of an individual post. Discarding thread %s.', thread_id)
                return None
            if post_username not in user_to_agent:
                speaker = self.determine_speaker(formatted_post, post_user_id, post_username, shortname)
                agents[speaker.name] = speaker
                if len(speaker.name) > 30:
                    self.logger.error('Invalid speaker. Skipping episode thread %s.', thread_id)
                    return None
                user_to_agent[post_username] = speaker.name
            speaker_name = user_to_agent[post_username]
            if len(formatted_post) > self.settings['max_utterance_chars']:
                sub_utterances = right_size(scenes=[formatted_post],
                                            max_length=self.settings['max_utterance_chars'])
                for sub_utterance in sub_utterances:
                    posts.append((speaker_name, sub_utterance))
            else:
                posts.append((speaker_name, formatted_post))
        return EnjimEpisode(forum_shortname=shortname, thread_subject=rp_thread_subject, posts=posts, agents=agents,
                            thread_id=thread_id)

    @lru_cache(maxsize=10_000)
    def get_chardef(self, user_id, char_name, shortname) -> t.Optional[EnjimAgent]:
        self.load_sqlite()
        self.logger.info('Making/fetching character named %s with user id %s for forum %s.', user_id, char_name,
                         shortname)
        char_forums = self.settings['sources'][shortname]['character_forums']
        for thread_id, thread_subject, post_content, post_user_id, post_username in self.conns[shortname].execute(
                self.THREAD_SELECT, (", ".join(char_forums), user_id)):
            if ' List' not in thread_subject and (char_name in post_content or char_name in thread_subject):
                self.logger.info('Found character thread for %s.', char_name)
                persona = self.parser.to_markdown(post_content)
                character: EnjimAgent = EnjimAgent(name=self.parser.to_markdown(thread_subject),
                                                   user_name=post_username,
                                                   user_id=user_id,
                                                   persona=persona)
                return character
            self.logger.warning('Thread %s not a match for character %s.', thread_subject, char_name)
        self.logger.debug('No character found for user_id %s, character name %s, forum %s. Will not have a persona.',
                          user_id, char_name, shortname)

    def determine_speaker(self, formatted_post, post_user_id, post_username, shortname) -> EnjimAgent:
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
            speaker = EnjimAgent(name=self.parser.to_markdown(post_username),
                                 user_name=post_username, user_id=post_user_id,
                                 persona='')
        return speaker

    def load_sqlite(self, force_recreate=False):
        """
        Downloads the file(s) and sets up connection(s) to the database(s).
        """
        if not force_recreate and self.conns is not None and len(self.conns) > 0:
            return
        self.conns = {}
        for shortname, configuration in self.settings['sources'].items():
            path = configuration['path']
            self.conns[shortname] = setup_sqlite(self.root_data_path, shortname, path, logger=self.logger)
        # Duct tape tier way of dealing with how slow running image recognition and summarization pipelines is.
        self.conns[self.CACHE_DB] = setup_sqlite(self.root_data_path, 'cache', self.settings['cache_db'],
                                                 logger=self.logger)
        self.conns[self.CACHE_DB].execute(
            'CREATE TABLE IF NOT EXISTS img_cache '
            '(img_url TEXT NOT NULL, model TEXT NOT NULL, description TEXT NOT NULL, '
            'CONSTRAINT img_pkey PRIMARY KEY (img_url, model))')
        self.conns[self.CACHE_DB].execute(
            'CREATE TABLE IF NOT EXISTS summary_cache '
            '(forum_shortname TEXT, text_id TEXT, max_length integer, summary TEXT, '
            'CONSTRAINT summ_pkey PRIMARY KEY (forum_shortname, text_id, max_length))')


def setup_sqlite(root_data_path, shortname, url_path, logger=logging.getLogger()):
    data_path = f"{root_data_path}/{shortname}.db"
    if not isfile(data_path) and len(url_path) > 0:
        logger.info('Downloading dataset %s from %s.', shortname, url_path)
        dataset = requests.get(url_path)
        with open(data_path, 'wb') as f:
            f.write(dataset.content)
    try:
        return sqlite3.connect(data_path)
    except Exception as error:
        logger.error('Error while connecting with db %s, error: %s.', shortname, error)
        raise error
    finally:
        logger.info('Added db %s.', shortname)
