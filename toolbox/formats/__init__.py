from typing import Type

from ..core import BaseFormat
from .metharme import MetharmeFormat
from .openchat import OpenChatFormat
from .sharegpt import ShareGptFormat

# Make this more dynamic later.
NAME_TO_FORMAT_MAPPING: dict[str, Type[BaseFormat]] = {
    "metharme": MetharmeFormat,
    "openchat": OpenChatFormat,
    "sharegpt": ShareGptFormat,
}
