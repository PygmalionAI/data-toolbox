import json
import logging
import os
import typing as t
from dataclasses import dataclass

from toolbox.datasets import BaseDataset
from toolbox.utils.dataset import get_data_path

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class CoTEpisode:
    question: str
    answer: str
    chain_of_thought: str

class CoTDataset(BaseDataset[CoTEpisode]):
    '''
    Chain of thought reasoning taken from the FLAN repository.
    Though the original repo has the CoT data in .tsv format, our dataset has converted it
    to .jsonl files.

    https://github.com/google-research/FLAN
    '''

    def generator(self) -> t.Generator[CoTEpisode, None, None]:
        # Go through CoT files
        for data in _available_jsonl_data():
            # Parse out options from the question.

            yield CoTDataset(
                question=data['question'],
                answer=data['answer'],
                chain_of_thought=data['chain_of_thought']
            )

# Private helpers

def _enumerate_jsonl_files(root_path: str) -> list[str]:
    '''Returns a list of files available in the given `root_path`.'''
    items = os.listdir(root_path)

    files: list[str] = []
    for item in items:
        item_path = os.path.join(root_path, item)
        if not os.path.isfile(item_path) or not item_path.endswith(".jsonl"):
            # We only care about JSONL files.
            continue

        absolute_file_path = os.path.abspath(item_path)
        files.append(absolute_file_path)
    
    return files

def _available_jsonl_data() -> t.Generator[dict[str, t.Any], None, None]:
    '''
    Yields all available JSONL data, parsed from the files in the Chain of Thought
    data folder.
    '''
    dataset_path = get_data_path(dataset_name="cot")
    for jsonl_file_path in _enumerate_jsonl_files(dataset_path):
        with open(jsonl_file_path, "r", encoding="utf-8") as jsonl_file:
            try:
                for line in jsonl_file:
                    yield json.loads(line)
            except json.decoder.JSONDecodeError as ex:
                logger.error("Failed to parse %s: $s", jsonl_file_path, ex)
