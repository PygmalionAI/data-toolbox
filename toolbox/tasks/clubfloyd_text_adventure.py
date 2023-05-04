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
    '''Instruction following task based on the evol_instruct (WizardLM) data.'''

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
    '''%{This is|You are|Start|Simulate|You are to simulate|Begin} a text %{adventure|adventure game} about the following:

{{description}}.

{{discretion_advised_str}}. %{Include|Incorporate|Use|Respect} the following %{themes|tags|concepts|genres|styles}: {{tags}}'''
])

SFW_PROMPTS = generate_prompts([
    "%{Generations|Your writing|The generated response|Your reply|Generated replies} must %{be safe for work|be SFW|not include any adult themes|be safe for minors|not include 18+ content|not be 18+|not be NSFW}",
])

NSFW_PROMPTS = generate_prompts([
    "%{Generations|Your writing|The generated response|Your reply|Generated replies} must %{be not safe for work|be NSFW|include adult themes|include erotic themes|include 18+ content}",
])