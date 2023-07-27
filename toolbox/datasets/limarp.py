# Much of this taken from dataprepare.py in the LIMARP, thanks anon
# If it ain't broke, don't fix it!
import glob
import os
import typing as t
import yaml

from dataclasses import dataclass

from toolbox.core.dataset import BaseDataset, get_path_for

@dataclass(frozen=True)
class LimaRpEntry:
    personas: dict[str, str]
    names: dict[str, str]
    scenario: str
    conversation: list[dict[str, str]]
    forum: str
    thread_id: int

class LimaRpDataset(BaseDataset[LimaRpEntry]):
    '''A collection of high-quality hand-curated roleplays.'''
    def __iter__(self) -> t.Generator[LimaRpEntry, None, None]:
        base_path = get_path_for("lima-erp")
        glob_path = f"{os.path.normpath(base_path)}/data/**/*.yaml"
        file_paths = glob.glob(glob_path, recursive=True)

        for file in file_paths:
            forum = os.path.basename(os.path.dirname(file))
            thread_id = os.path.basename(file).split(".")[0]
            with open(file, 'r', encoding='utf-8') as f:
                source = yaml.safe_load(f)
                yield LimaRpEntry(
                    personas=source["personas"],
                    names=source["names"],
                    scenario=source["scenario"],
                    conversation=source["conversation"],
                    forum=forum,
                    thread_id=thread_id,
                )
