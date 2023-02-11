'''Utility functions to clean up text strings.'''

# Some of this is pasta from Meta's ParlAI. See:
# https://github.com/facebookresearch/ParlAI/blob/main/parlai/utils/strings.py


def normalize_string(text: str, add_trailing_period: bool = True) -> str:
    '''
    Standardize the capitalization and punctuation spacing of the input text.
    - Version 1: Fix sentence start casing and punctuation.
    - Version 2: Add trailing period, if missing.
    '''

    switch_list = [(' .', '.'), (' ,', ','), (' ?', '?'), (' !', '!'),
                   (" ' ", "'")]

    # add spaces so that words and punctuation can be seaprated
    new_text = text.lower()

    # normalize in case of human:
    for new, old in switch_list:
        new_text = new_text.replace(old, new).replace('  ', ' ')

    # split on punctuation to find sentence boundaries
    # capitalize stuff
    tokens = new_text.split(' ')
    for i in range(len(tokens)):
        if i == 0:
            tokens[i] = uppercase(tokens[i])
        elif tokens[i] in ('i', "i'm", "i've", "i'll", "i'd"):
            tokens[i] = uppercase(tokens[i])
        elif tokens[i] in '?.!' and i < len(tokens) - 1:
            tokens[i + 1] = uppercase(tokens[i + 1])
    new_text = ' '.join(tokens)
    new_text = ' ' + new_text + ' '

    for tup in switch_list:
        new_text = new_text.replace(tup[0], tup[1])

    # get rid of surrounding whitespace
    new_text = new_text.strip()
    new_text = new_text.replace('  ', ' ')

    if add_trailing_period and new_text and new_text[-1] not in '!.?)"\'':
        new_text += '.'

    return new_text


def title_case(string: str) -> str:
    '''Converts a string into Title Case.'''
    return " ".join([uppercase(word) for word in string.split(" ")])


def uppercase(string: str) -> str:
    '''
    Makes the first character of the string uppercase, if the string is
    non-empty.
    '''
    if len(string) == 0:
        return string
    else:
        return string[0].upper() + string[1:]
