import os
import typing as t
from os.path import isfile

import yaml

HERE = os.path.realpath(os.path.dirname(__file__))


def get_data_path(dataset_name: t.Optional[str] = None) -> str:
    """
    Returns an absolute path to either the data folder, or a specific dataset if
    `dataset_name` is supplied.
    """
    if 'WAIFU_DATA_PATH' in os.environ:
        return os.environ['WAIFU_DATA_PATH']

    components = [HERE, "..", "..", "data"]
    if dataset_name:
        components.append(dataset_name)

    path = os.path.join(*components)
    os.makedirs(path, exist_ok=True)
    return path


def get_config(dataset_name: t.Optional[str] = None, filename: str = 'resources/config.yaml'):
    if dataset_name is None or not isfile(filename):
        return {}
    with open(filename, 'r') as configuration_file:
        return yaml.safe_load(configuration_file)[dataset_name]
