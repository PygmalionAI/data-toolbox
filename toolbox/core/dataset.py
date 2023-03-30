import os
import typing as t

HERE = os.path.realpath(os.path.dirname(__file__))
T = t.TypeVar("T")


class BaseDataset(t.Generic[T]):
    '''Base dataset class.'''

    def __iter__(self) -> t.Generator[T, None, None]:
        '''
        This method must be overidden when inheriting. It should yield
        individual items from the dataset.
        '''
        raise NotImplementedError


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
        components = [HERE, "..", "..", "data"]

    if dataset_name is not None:
        components.append(dataset_name)

    return os.path.join(*components)
