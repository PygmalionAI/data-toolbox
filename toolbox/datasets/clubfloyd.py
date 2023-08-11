import json
import os
import typing as t
from dataclasses import dataclass

from toolbox.core.dataset import BaseDataset, get_path_for


@dataclass(frozen=True)
class StoryAction:
    action: str
    response: str
    endoftext: bool


@dataclass(frozen=True)
class ClubFloydStory:
    name: str
    author: str
    genres: list[str]
    tags: list[str]
    year: int
    ratings: list[int]
    total_ratings: int
    average_rating: float
    transcript_id: str
    discretion_advised: bool
    description: str
    actions: list[StoryAction]


class ClubFloydDataset(BaseDataset[ClubFloydStory]):
    '''
    Data from VE's ClubFloyd scrape.

    https://wandb.ai/ve-forbryderne/skein/runs/files/files/datasets/floyd
    '''

    def __iter__(self) -> t.Generator[ClubFloydStory, None, None]:
        root_path = get_path_for("club-floyd")
        file_path = os.path.join(root_path, "floyd.json")

        with open(file_path, "r") as file:
            raw_stories = json.load(file).values()
            for raw_story in raw_stories:
                actions = [
                    _story_action_from_dict(action)
                    for action in raw_story["data"]
                ]

                yield ClubFloydStory(
                    name=raw_story["name"],
                    author=raw_story["author"],
                    genres=raw_story["genres"],
                    tags=raw_story["tags"],
                    year=raw_story["year"],
                    ratings=raw_story["ratings"],
                    total_ratings=raw_story["total_ratings"],
                    average_rating=raw_story["average_rating"],
                    transcript_id=raw_story["transcript_id"],
                    discretion_advised=raw_story["discretion_advised"],
                    description=raw_story["description"],
                    actions=actions,
                )


def _story_action_from_dict(data: dict[str, str | bool]) -> StoryAction:
    action = data["action"]
    response = data["response"]
    endoftext = data["endoftext"]

    assert isinstance(action, str), "Unexpected type for `action` field"
    assert isinstance(response, str), "Unexpected type for `response` field"
    assert isinstance(endoftext, bool), "Unexpected type for `endoftext` field"

    return StoryAction(
        action=action,
        response=response,
        endoftext=endoftext,
    )
