import re
import typing as t

# The regex used to find message variants (e.g.: `%{Hi|Hello} there!`)
VARIANT_REGEX = re.compile(r'%{(.+?)}')


def generate_variants_for(
        string: str,
        max_generations: int | None = 256,
        start_counter_at: int = 0) -> t.Generator[str, None, None]:
    '''
    Given a string like "%{Hello|Hi} there%{.|!}, this should yield:

    - Hello there.
    - Hello there!
    - Hi there.
    - Hi there!
    '''

    # Some bot creators went wild with the variants, which causes ridiculous
    # generations if we try to exhaust all possibilities so we cap that here.
    # `start_counter_at` is used for keeping track across recursive calls.
    counter = start_counter_at

    if (match := re.search(VARIANT_REGEX, string)) is not None:
        # Once we have a "%{X|Y|Z}" matched inside the original string, we:
        # - Fetch .groups()[0] (which will give us `X|Y|Z`)
        # - Split by `|` (so we have ["X", "Y", "Z"])
        # - Filter out empty strings
        alternatives = filter(lambda x: x.strip(), match.groups()[0].split("|"))

        # Then, we break the string apart into what comes before and after the
        # alternatives, that way we can re-build with "prefix + choice + sufix".
        prefix = string[:match.start()]
        sufix = string[match.end():]

        for alternative in alternatives:
            variant = f'{prefix}{alternative}{sufix}'

            # However, some strings have multiple variant blocks. In that case,
            # we operate on them recursively until we have just regular strings
            # after generating all possible variants.
            still_have_match = re.search(VARIANT_REGEX, variant) is not None
            if still_have_match:
                for inner_variant in generate_variants_for(
                        variant, start_counter_at=counter):
                    yield inner_variant

                    # Keep track and break after `max_generations`.
                    counter += 1
                    if max_generations is not None and counter >= max_generations:
                        break
            else:
                yield variant

                # Keep track and break after `max_generations`.
                counter += 1
                if max_generations is not None and counter >= max_generations:
                    break
    else:
        yield string


def generate_prompts(system_prompts: list[str]) -> list[str]:
    '''
    Given a list of base system prompts,
    this function generates a list of variants on these prompts using generate_variants_for
    '''
    unflattened_list = [list(generate_variants_for(x)) for x in system_prompts]

    flattened_list: list[str] = []
    for l in unflattened_list:
        flattened_list += l

    return flattened_list
