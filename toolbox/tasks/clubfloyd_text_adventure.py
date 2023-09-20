import logging
import random
import typing as t

from toolbox.core.models import Episode, Turn, TurnKind
from toolbox.core.task import BaseTask
from toolbox.datasets.clubfloyd import ClubFloydDataset
from toolbox.utils.prompts import generate_prompts

LOG = logging.getLogger(__name__)

MIN_USER_RATING = 3.0


class ClubFloydTextAdventureTask(BaseTask):
    '''Text adventure task based on ClubFloyd data.'''

    def __iter__(self) -> t.Generator[Episode, None, None]:
        for idx, story in enumerate(ClubFloydDataset()):
            if story.average_rating < MIN_USER_RATING:
                # Kills off ~15% of the data IIRC, so this feels like a nice
                # trade-off.
                continue

            sp = random.choice(_SYSTEM_PROMPTS)
            sp = sp.replace("{{title}}", story.name)
            sp = sp.replace("{{description}}", story.description)
            sp = sp.replace(
                "{{discretion_advised_str}}",
                random.choice(
                    NSFW_PROMPTS if story.discretion_advised else SFW_PROMPTS))
            sp = sp.replace("{{tags}}",
                            _process_tags(story.tags + story.genres))

            turns: list[Turn] = [
                Turn(utterance=sp, kind=TurnKind.SYSTEM),
            ]

            for action in story.actions:
                # If the user's input is just `%` that means "start the game".
                # We don't want to require that at inference time, so let's just
                # skip straight to the game starting.
                if action.action == "%":
                    turns.append(
                        Turn(utterance=action.response, kind=TurnKind.MODEL))
                else:
                    user_turn = Turn(utterance=action.action,
                                     kind=TurnKind.USER)
                    model_turn = Turn(utterance=action.response,
                                      kind=TurnKind.MODEL)

                    turns += [user_turn, model_turn]

            yield Episode(turns=turns, identifier=f"club-floyd-{idx}")


def _process_tags(tags: list[str]) -> str:
    tags = [
        tag for tag in tags if all([
            # Filter out tags according to these criteria.
            word not in tag.lower() for word in [
                "steam",
                "collaboration",
                "cover art",
                "inform 7",
                "walkthrough",
                "parser",
                "many authors",
            ]
        ])
    ]

    # Shuffle and remove duplicates to ensure data diversity.
    tags = list(set(tags))
    random.shuffle(tags)

    return ", ".join(tags)


_SYSTEM_PROMPTS = generate_prompts([
    '''%{This is|You are|Start|Simulate|You are to simulate|Begin} a text %{adventure|adventure game} %{in the style of|similar to|like} {{title}}. {{discretion_advised_str}}.

%{Include|Incorporate|Use|Respect} the following %{themes|tags|concepts|genres|styles}: {{tags}}''',
#
    '''%{This is|You are|Start|Simulate|You are to simulate|Begin} a text %{adventure|adventure game} about the following:

{{description}}.

{{discretion_advised_str}}. %{Include|Incorporate|Use|Respect} the following %{themes|tags|concepts|genres|styles}: {{tags}}''',
# No tags so model can learn to diversify content without explicit prompting
'''%{Here|The following paragraph|The upcoming paragraph|The following} is %{a description|an overview} of a %{text game|text RPG|text adventure|text adventure game} %{called|named} {{title}}.
Its %{description|synopsis} is %{the following|as follows}:
{{description}}
Be sure to drive the story forward.''',
#
'''I am to %{generate|write|engage in} a %{text adventure|CYOA-style game|creative text RPG|text adventure game} with the following %{tags|themes|genres}: {{tags}}
Here is %{the description of the game|what the game is about}: {{description}}.''',
#
'''%{Mode|Current mode}: %{text adventure|dungeon master|DM|adventure game in text form}
%{Description|Overview}: {{description}}
%{Tags|Genres}: {{tags}}''',
'''%{Enter|Engage|Consider} %{game|adventure game|text adventure|text RPG} mode. %{Here|In this mode}, you will respond to the user's %{commands|prompts} and drive %{a|the} %{story|plot} %{forward|forwards}.'''
# Just the length prompt
'''{{response_length_str}}.''',
# basic
'''text game''',
# Nothing
''''''
])

SFW_PROMPTS = generate_prompts([
    "%{Generations|Your writing|The generated response|Your reply|Generated replies} must %{be safe for work|be SFW|not include any adult themes|be safe for minors|not include 18+ content|not be 18+|not be NSFW}",
])

NSFW_PROMPTS = generate_prompts([
    "%{Generations|Your writing|The generated response|Your reply|Generated replies} must %{be not safe for work|be NSFW|include adult themes|include erotic themes|include 18+ content}",
])