import os
import pickle
import typing as t
from dataclasses import dataclass

import mashumaro
import pandas as pd

from toolbox.datasets import BaseDataset
from toolbox.utils.dataset import get_data_path


@dataclass(frozen=True)
class SodaEpisode(mashumaro.DataClassDictMixin):
    narrative: str
    dialogue: t.List[str]
    speakers: t.List[str]
    relation: str
    literal: str


class SodaDataset(BaseDataset[SodaEpisode]):
    '''
    SODA: Million-scale Dialogue Distillation with Social Commonsense
    Contextualization

    https://huggingface.co/datasets/allenai/soda
    '''

    def generator(self) -> t.Generator[SodaEpisode, None, None]:
        root_data_path = get_data_path("soda")
        file_path = os.path.join(root_data_path, "test.parquet")
        df = pd.read_parquet(file_path)

        # Iterate through the test part of the SODA dataset
        for i in df.index:
            yield SodaEpisode(narrative=df['narrative'][i],
                              dialogue=df['dialogue'][i],
                              speakers=df['speakers'][i],
                              relation=df['relation'][i],
                              literal=df['literal'][i])
