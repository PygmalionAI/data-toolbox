import os
import logging

from toolbox.utils.dataset import get_data_path

LOG = logging.getLogger(__name__)


def enumerate_dataset_files(datset_name: str,
                            file_extensions: list[str]) -> list[str]:
    '''Returns a list of files available for the given dataset.'''
    dataset_path = get_data_path(dataset_name=datset_name)
    items = os.listdir(dataset_path)

    files: list[str] = []
    for item in items:
        item_path = os.path.join(dataset_path, item)
        if not os.path.isfile(item_path):
            # We don't care about folders.
            continue

        if not any([item_path.endswith(ext) for ext in file_extensions]):
            # Ignore invalid file extensions.
            continue

        absolute_file_path = os.path.abspath(item_path)
        files.append(absolute_file_path)

    return files
