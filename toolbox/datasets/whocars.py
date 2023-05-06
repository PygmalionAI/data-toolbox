import csv
import json
import logging
import typing as t
from dataclasses import dataclass

from toolbox.core.dataset import BaseDataset
from toolbox.utils.files import enumerate_files_for

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class WhocarsEntry:
    model: str
    endpoint: str
    prompt_json: dict[str, t.Any]
    response: str


class WhocarsDataset(BaseDataset[WhocarsEntry]):
    '''Logs from the whocars proxy.'''

    def __iter__(self) -> t.Generator[WhocarsEntry, None, None]:
        for file_path in enumerate_files_for("whocars", file_extension=".csv"):
            if "__index__" in file_path:
                continue

            with open(file_path, "r") as file:
                reader = csv.DictReader(file)
                try:
                    for row in reader:
                        yield WhocarsEntry(
                            model=row["model"],
                            endpoint=row["endpoint"],
                            prompt_json=json.loads(row["prompt json"]),
                            response=row["response"],
                        )
                except csv.Error as ex:
                    # One file seems to have broken encoding, just skip over it,
                    # we have enough data otherwise.
                    LOG.error(ex)