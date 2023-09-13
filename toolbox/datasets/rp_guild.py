import ast
import csv
import logging
import sys
import typing as t

from dataclasses import dataclass

from toolbox.core.dataset import BaseDataset
from toolbox.datasets.rp_forums import RpMessage
from toolbox.utils.files import enumerate_files_for

LOG = logging.getLogger(__name__)

@dataclass(frozen=True)
class RpGuildThread:
    messages: list[RpMessage]
    thread_name: str
    thread_type: str
    tags: list[str]

class RpGuildDataset(BaseDataset[RpGuildThread]):
    """Data scraped from the Roleplayers Guild forum."""
    def __iter__(self) -> t.Generator[RpGuildThread, None, None]:
        # NOTE(TG): If csv fields are longer than 131,072 characters,
        # the csv library shits itself by default, so we fix that here.
        # See note from 11b in rp_forums.py for further details.
        csv.field_size_limit(sys.maxsize)
        for path in enumerate_files_for(dataset_name="rp-guild", file_extension=".csv"):
            with open(path, "r") as file:
                reader = csv.DictReader(file, delimiter=",")

                # Store a buffer of the previous thread
                previous_thread = None
                previous_type = None
                previous_tags = None
                previous = [previous_thread, previous_type, previous_tags]

                current_thread = None
                current_type = None
                current_tags = None
                messages: list[RpMessage] = []
                
                for row in reader:
                    if row['thread_title'] != previous_thread or row['thread_type'] != previous_type:
                        if len(messages) != 0:
                            # Ugly assertion, but it'll do
                            #print(messages)
                            #print(previous_thread, previous_type, previous_tags)
                            #assert all([(lambda x: x is not None)(b) for b in previous])
                            # Yield the thread with the buffer
                            yield RpGuildThread(
                                messages=messages,
                                thread_name=previous_thread,
                                thread_type=previous_type,
                                tags=previous_tags,
                            )

                        # Update buffer now that the thread is yielded
                        previous_type = current_type
                        previous_thread = current_thread
                        previous_tags = current_tags
                        messages = []

                    current_thread = row['thread_title']
                    current_type = row['thread_type']
                    # Do safe eval here to convert a string of a list into a proper list
                    # without having to do a bunch of parsing
                    current_tags = ast.literal_eval(row['thread_tags'])

                    # Necessary to avoid weird errors? I dunno, it's 12 AM.
                    # TODO(TG): Fix this. All of this.
                    if any((lambda x: x is None)(x) for x in previous):
                        previous_type = current_type
                        previous_thread = current_thread
                        previous_tags = current_tags

                    message = RpMessage(author=row['message_username'], message=row['message'])
                    messages.append(message)
