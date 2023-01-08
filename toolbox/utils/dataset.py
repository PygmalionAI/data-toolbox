import os
import typing as t

HERE = os.path.realpath(os.path.dirname(__file__))


def get_data_path(dataset_name: t.Optional[str] = None) -> str:
    '''
    Returns an absolute path to either the data folder, or a specific dataset if
    `dataset_name` is supplied.
    '''
    if 'WAIFU_DATA_PATH' in os.environ:
        return os.environ['WAIFU_DATA_PATH']

    components = [HERE, "..", "..", "data"]
    if dataset_name:
        components.append(dataset_name)

    return os.path.join(*components)
