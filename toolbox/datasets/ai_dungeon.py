import os
import typing as t

from toolbox.core.dataset import BaseDataset, get_path_for


class AiDungeonDataset(BaseDataset[str]):
    '''
    AI Dungeon's `text_adventures.txt`.
    '''

    def __iter__(self) -> t.Generator[str, None, None]:
        root_path = get_path_for("ai-dungeon")
        file_path = os.path.join(root_path, "text_adventures.txt")

        with open(file_path, "r") as file:
            for line in file:
                yield line
