import csv
import os
import sys
import typing as t
from dataclasses import dataclass

from toolbox.core.dataset import BaseDataset, get_path_for


@dataclass(frozen=True)
class McStory:
    title: str
    author: str
    date: str
    tags: str
    summary: str
    href: str
    header: str
    text_contents: str
    footer: str


class McStoriesDataset(BaseDataset[McStory]):
    '''Data from a certain story-sharing site.'''

    def __iter__(self) -> t.Generator[McStory, None, None]:
        # NOTE(11b): I had no idea this was a thing, but apparently Python's CSV
        # reader by default shits the bed if you have a field longer than 131072
        # characters. _Usually_ this means you've messed up the parsing, but in
        # our case it's actually just a massive forum post triggering this.
        # https://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072
        csv.field_size_limit(sys.maxsize)

        root_data_path = get_path_for("mcstories")
        file_path = os.path.join(root_data_path, "mcstories--all.csv")

        with open(file_path, "r") as file:
            reader = csv.DictReader(file, delimiter=",")
            for row in reader:
                story = McStory(
                    title=row["story_title"],
                    author=row["story_author"],
                    date=row["story_date"],
                    tags=row["story_tags"],
                    summary=row["story_summary"],
                    href=row["story_href"],
                    header=row["story_header"],
                    text_contents=row["story"],
                    footer=row["story_footer"],
                )
                yield story