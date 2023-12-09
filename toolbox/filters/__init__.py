from typing import Type

from .applefilter import AppleFilter
from .english import EnglishFilter
from .exact_deduplication import ExactDedupFilter
from .llmslop import LlmSlopFilter
from .low_quality_rp import LowQualityRpFilter
from .refusals import RefusalFilter
from .similarity import SimilarityFilter
from ..core import BaseFilter

NAME_TO_FILTER_MAPPING: dict[str, Type[BaseFilter]] = {
    cls.__name__: cls for cls in [
        AppleFilter,
        EnglishFilter,
        ExactDedupFilter,
        LlmSlopFilter,
        LowQualityRpFilter,
        RefusalFilter,
        SimilarityFilter
    ]
}
