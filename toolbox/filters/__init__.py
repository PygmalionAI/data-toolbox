from .apple import AppleFilter
from .naive_refusal import NaiveRefusalFilter

# Mapping of filter shorthand names to their corresponding classes.
# Note that simply specifying the name of the filter class should also work.
FILTER_MAPPINGS = {
    (cls.__name__, cls.FILTER_SHORTHAND): cls for cls in [
        AppleFilter,
        NaiveRefusalFilter,
    ]
}
