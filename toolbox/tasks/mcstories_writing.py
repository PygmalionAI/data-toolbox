import logging
import random
import typing as t

from markdownify import markdownify

from toolbox.core.models import Episode, Turn, TurnKind
from toolbox.core.task import BaseTask
from toolbox.datasets.mcstories import McStoriesDataset
from toolbox.utils.prompts import generate_prompts

LOG = logging.getLogger(__name__)


class McStoriesWritingTask(BaseTask):
    '''Story-writing task based on McStories data.'''

    def __iter__(self) -> t.Generator[Episode, None, None]:
        for idx, story in enumerate(McStoriesDataset()):

            contents = _html_story_to_clean_md(story.text_contents)
            chunks = _split_text_into_chunks(contents, min_word_count=250)

            # Compose synthetic the system prompt.
            system_prompt = random.choice(_SYSTEM_PROMPTS)
            system_prompt = system_prompt.replace("{{title}}", story.title)
            system_prompt = system_prompt.replace("{{summary}}", story.summary)

            full_tags = [
                _TAG_SHORTHANDS_TO_FULL_MAPPING[shorthand]
                for shorthand in story.tags[1:-1].replace("'", "").split(", ")
            ]
            system_prompt = system_prompt.replace("{{tags}}",
                                                  ", ".join(full_tags))

            turns: list[Turn] = [
                Turn(utterance=system_prompt, kind=TurnKind.SYSTEM)
            ]

            for chunk in chunks:
                turns.append(Turn(
                    utterance=chunk,
                    kind=TurnKind.MODEL,
                ))

            yield Episode(turns=turns, identifier=f"mcstories-{idx}")


def _html_story_to_clean_md(html: str) -> str:
    md = str(markdownify(html))

    lines: list[str] = []
    for line in md.splitlines():
        # These usually denote chapter titles, or author names/emails which we
        # don't want the model learning.
        if line.startswith("###"):
            continue
        lines.append(line.strip())

    return "\n".join(lines)


def _split_text_into_chunks(text: str, min_word_count: int) -> list[str]:
    '''
    Breaks `text` apart into paragraphs, then joins up paragraphs until they
    reach `min_word_count`.
    '''
    output: list[str] = []
    paragraphs = text.split("\n\n")
    acc = ""

    for paragraph in paragraphs:
        acc += f"\n\n{paragraph}"
        if len(acc.split()) > min_word_count:
            output.append(acc.strip())
            acc = ""

    return output


_BASE_SYSTEM_PROMPTS = [
    '''You %{are to|should|must|will now} %{generate|write} a %{story|fictional story}. Its title should be "{{title}}", and it should %{include|adhere to|contain} the following themes: {{tags}}. {{response_length_str}}. %{The story should be about|Summary|Quick rundown|It's about|Theme|Contents}: {{summary}}''',
    '''You %{are to|should|must|will now} %{generate|write} a %{story|fictional story} titled "{{title}}". It should %{include|adhere to|contain} the following themes: {{tags}}. %{The story should be about|Summary|Quick rundown|It's about|Theme|Contents}: {{summary}}. {{response_length_str}}.''',
    '''{{response_length_str}}. You %{are to|should|must|will now} %{generate|write} a %{story|fictional story}. %{The story should be about|Summary|Quick rundown|It's about|Theme|Contents}: {{summary}}. Include the following %{themes|tags}: {{tags}}.''',
]

_SYSTEM_PROMPTS = generate_prompts(_BASE_SYSTEM_PROMPTS)

_TAG_SHORTHANDS_TO_FULL_MAPPING = {
    'bd': 'bondage and/or discipline',
    'be': 'bestiality',
    'ca': 'cannibalism',
    'cb': 'comic book super-hero/heroine',
    'ds': 'dominance and/or submission',
    'ex': 'exhibitionism',
    'fd': 'female dominant',
    'ff': 'female/female sex',
    'ft': 'fetish clothing',
    'fu': 'furry',
    'gr': 'growth/enlargement',
    'hm': 'humiliation',
    'hu': 'humor',
    'in': 'incest',
    'la': 'lactation',
    'ma': 'masturbation',
    'mc': 'mind control',
    'md': 'male dominant',
    'mf': 'male/female sex',
    'mm': 'male/male sex',
    'nc': 'non-consensual',
    'rb': 'robots',
    'sc': 'scatology',
    'sf': 'science fiction',
    'ts': 'time stop',
    'ws': 'watersports',
}