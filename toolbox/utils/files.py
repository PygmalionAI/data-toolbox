import logging
import os

from toolbox.utils.dataset import get_data_path

LOG = logging.getLogger(__name__)


def enumerate_dataset_files(
    datset_name: str,
    file_extension: str,
    subfolder: str | None = None,
) -> list[str]:
    '''Returns a list of files available for the given dataset.'''
    dataset_path = get_data_path(dataset_name=datset_name)
    final_path = dataset_path if subfolder is None else os.path.join(
        dataset_path, subfolder)
    items = os.listdir(final_path)

    files: list[str] = []
    for item in items:
        item_path = os.path.join(final_path, item)
        if not os.path.isfile(item_path):
            # We don't care about folders.
            continue

        if not item_path.endswith(file_extension):
            # Ignore invalid file extensions.
            continue

        absolute_file_path = os.path.abspath(item_path)
        files.append(absolute_file_path)

    return files
