import typing as t
from dataclasses import dataclass

import pandas as pd

from toolbox.core.dataset import BaseDataset
from toolbox.utils.files import enumerate_files_for


@dataclass(frozen=True)
class Gpt4AllDataInstance:
    prompt: str
    response: str
    source: str


class Gpt4AllDataset(BaseDataset[Gpt4AllDataInstance]):
    '''
    NomicAI's GPT4all dataset.

    https://huggingface.co/datasets/nomic-ai/gpt4all_prompt_generations
    '''

    def __iter__(self) -> t.Generator[Gpt4AllDataInstance, None, None]:
        parquet_files = enumerate_files_for("gpt4all_prompt_generations",
                                            file_extension="parquet")

        for file in parquet_files:
            df = pd.read_parquet(file)
            for idx in df.index:
                yield Gpt4AllDataInstance(
                    prompt=df["prompt"][idx],
                    response=df["response"][idx],
                    source=df["source"][idx],
                )
