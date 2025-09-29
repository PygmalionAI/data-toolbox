from .apple import AppleFilter

# Mapping of filter shorthand names to their corresponding classes.
# Note that simply specifying the name of the filter class should also work.
FILTER_MAPPINGS = {
    ("AppleFilter", "apple_filter"): AppleFilter,
}
