'''Common functions for processing text.'''
import re

import ftfy

EXCESSIVE_CHARS_PATTERN = re.compile(r"(\.|\-|\*|!)\1{3,}")

# Custom ftfy config to keep stuff such as full-width text (for fun!)
config = ftfy.TextFixerConfig(
    fix_latin_ligatures=False,
    fix_character_width=False,
    # We don't need explanations for now and extra speed is wanted.
    explain=False
)

def fix_style_and_encoding_issues(original_message: str) -> str:
    '''Cleans up any style-related issues.'''
    message = original_message
    message = message.replace(" .. ", "... ")
    message = message.replace(" ... ", "... ")
    message = re.sub(r'\b(\.\.\.?)\b', '... ', message)

    message = message.replace(" . ", ". ")
    message = message.replace(" , ", ", ")
    message = message.replace(" ? ", "? ")
    message = message.replace(" ! ", "! ")

    message = re.sub(r"(\S)(…|\.\.\.)(\S)", "\\1\\2 \\3", message)
    message = EXCESSIVE_CHARS_PATTERN.sub(r"\1\1\1", message)

    # Fix hidden zero-width spaces and other whitespace fuckery which could
    # mess with a model.
    document = document.replace(" ", "")
    document = document.replace("​", "")
    document = document.replace("‍", " ")
    document = document.replace(" ", " ")
    document = document.replace("﻿", " ")
    document = document.replace("", "")
    document = document.replace("‎", "")

    # Some forums have their pages incorrectly tagged as UTF-8, so we get
    # garbage when decoding. Most common problem I've seen is bad quotation
    # marks, so we paper over that here.
    message = ftfy.fix_text(message, config=config)

    return message
