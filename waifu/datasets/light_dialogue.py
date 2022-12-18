import os
import pickle
import typing as t
from dataclasses import dataclass

import mashumaro

from waifu.datasets import BaseDataset
from waifu.utils.dataset import get_data_path


@dataclass(frozen=True)
class LightDialogueAgent(mashumaro.DataClassDictMixin):
    name: str
    persona: str


@dataclass(frozen=True)
class LightDialogueSetting(mashumaro.DataClassDictMixin):
    name: str
    category: str
    description: str
    background: str


@dataclass(frozen=True)
class LightDialogueEpisode(mashumaro.DataClassDictMixin):
    agents: t.List[LightDialogueAgent]
    setting: LightDialogueSetting
    character: t.List[str]
    context: t.List[str]
    room_objects: t.List[t.List[str]]
    room_agents: t.List[t.List[str]]
    all_descriptions: t.Dict[str, str]
    available_actions: t.List[t.List[str]]
    carrying: t.List[t.List[str]]
    wielding: t.List[t.List[str]]
    speech: t.List[str]
    emote: t.List[str]
    action: t.List[str]


class LightDialogueDataset(BaseDataset[LightDialogueEpisode]):
    '''
    LIGHT: Learning in Interactive Games with Humans and Text

    The LIGHT project is a large-scale fantasy text adventure game research
    platform for training agents that can both talk and act, interacting either
    with other models or with humans.

    https://parl.ai/projects/light/
    '''

    def generator(self) -> t.Generator[LightDialogueEpisode, None, None]:
        root_data_path = get_data_path("light_dialogue")
        light_data_path = os.path.join(root_data_path, "light_data.pkl")

        with open(light_data_path, "rb") as light_data_file:
            light_data = pickle.load(light_data_file)
            for episode in light_data:
                yield LightDialogueEpisode.from_dict(episode)
