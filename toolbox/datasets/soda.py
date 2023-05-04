import os
import typing as t
from dataclasses import dataclass

import pandas as pd

from toolbox.core.dataset import BaseDataset, get_path_for


@dataclass(frozen=True)
class SodaEpisode:
    narrative: str
    dialogue: t.List[str]
    speakers: t.List[str]
    relation: str
    literal: str
    original_index: str


class SodaDataset(BaseDataset[SodaEpisode]):
    '''
    SODA: Million-scale Dialogue Distillation with Social Commonsense
    Contextualization

    https://huggingface.co/datasets/allenai/soda
    '''

    def __init__(self, split: str = "train") -> None:
        assert split in ["test", "train", "valid"]
        root_data_path = get_path_for("soda")
        self.file_path = os.path.join(root_data_path, f"{split}.parquet")

        super().__init__()

    def __iter__(self) -> t.Generator[SodaEpisode, None, None]:
        df = pd.read_parquet(self.file_path)
        for idx in df.index:
            yield SodaEpisode(narrative=df["narrative"][idx],
                              dialogue=df["dialogue"][idx],
                              speakers=df["speakers"][idx],
                              relation=df["relation"][idx],
                              literal=df["literal"][idx],
                              original_index=str(df["original_index"][idx]))
