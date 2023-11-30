from typing import Type

from .applefilter import AppleFilter
from ..core import BaseFilter

NAME_TO_FILTER_MAPPING: dict[str, Type[BaseFilter]] = {
    cls.__name__: cls for cls in [
        AppleFilter
    ]
}
