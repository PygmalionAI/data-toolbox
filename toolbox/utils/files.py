import logging
import os

DIRNAME = os.path.dirname # Shorten dirname for readability
HERE = os.path.realpath(os.path.dirname(__file__))
LOG = logging.getLogger(__name__)

def get_path_for(dataset_name: str | None) -> str:
    '''
    Returns an absolute path. If `dataset_name` is given, it will return the
    path to the specific dataset's folder, otherwise it'll return the path to
    the root data folder.
    '''

    # Allow overriding the location of the root data folder by using an
    # environment variable.
    env_var = "TOOLBOX_DATA_FOLDER"
    if env_var in os.environ:
        components = [os.environ[env_var]]
    else:
        # Repeatedly apply dirname to go 2 directory levels up
        # https://stackoverflow.com/questions/2817264/how-to-get-the-parent-dir-location
        components = [DIRNAME(DIRNAME(os.path.abspath(HERE))), "data"]

    if dataset_name is not None:
        components.append(dataset_name)

    return os.path.join(*components)

def enumerate_files_for(
    dataset_name: str,
    file_extension: str,
    subfolder: str | None = None,
) -> list[str]:
    '''Returns a list of files available for the given dataset.'''
    dataset_path = get_path_for(dataset_name)
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
