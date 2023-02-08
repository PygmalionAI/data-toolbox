import json
import logging
import typing as t
from dataclasses import dataclass

from toolbox.datasets import BaseDataset
from toolbox.utils.files import enumerate_dataset_files

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CoTEpisode:
    question: str
    answer: str
    chain_of_thought: str


class CoTDataset(BaseDataset[CoTEpisode]):
    '''
    Chain of thought reasoning taken from the FLAN repository.

    Though the original repo has the CoT data in .tsv format, our dataset has
    converted it to .jsonl files.

    https://github.com/google-research/FLAN
    '''

    def generator(self) -> t.Generator[CoTEpisode, None, None]:
        for path in enumerate_dataset_files("chain_of_thought",
                                            file_extensions=[".jsonl"]):
            with open(path, "r", encoding="utf-8") as file:
                for line in file:
                    data = json.loads(line)
                    yield CoTEpisode(question=data['question'],
                                     answer=data['answer'],
                                     chain_of_thought=data['chain_of_thought'])
