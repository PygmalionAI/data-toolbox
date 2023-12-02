'''Common functions for processing text.'''
import re

import ftfy

EXCESSIVE_CHARS_PATTERN = re.compile(r"(\.|\-|\*|!)\1{3,}")
EXTRA_NEWLINE_PATTERN = re.compile(r"\n{2,}")
LINKS_PATTERN = re.compile(r"https?:\/\/.+?(\s|$)")
MARKDOWN_IMAGE_EMBED_PATTERN = re.compile(r"!\[.*?\]\(.*?\)\n*")
# Gaze upon my works ye mighty and despair
MENTION_PATTERN = re.compile(r"(?<!\w)([^\S\r\n]|^)*@[^\W\s]+?(?=(,|\.|\?|~|!|\s|:|$))", flags=re.MULTILINE)
OOC_PATTERN = re.compile(r"((\[\[|\(\().*(\)\)|\]\])|\(OOC:.+\)|(?<=\s)OOC:.*(?!$))")
# Stop despairing now
UNSPACED_PUNCTUATION_PATTERN = re.compile(r"([a-z]{2,})(\.|!|\?)([A-Z])")

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

    message = EXTRA_NEWLINE_PATTERN.sub("\n\n", message)
    message = MARKDOWN_IMAGE_EMBED_PATTERN.sub("", message)

    message = re.sub(r"(\S)(…|\.\.\.)(\S)", "\\1\\2 \\3", message)
    message = EXCESSIVE_CHARS_PATTERN.sub(r"\1\1\1", message)

    message = message.replace(" .. ", "... ")
    message = message.replace(" ... ", "... ")
    message = re.sub(r'\b(\.\.\.?)\b', '... ', message)

    message = message.replace(" . ", ". ")
    message = message.replace(" , ", ", ")
    message = message.replace(" ? ", "? ")
    message = message.replace(" ! ", "! ")

    message = UNSPACED_PUNCTUATION_PATTERN.sub(r"\1\2 \3", message)

    # Fix hidden zero-width spaces and other whitespace fuckery which could
    # mess with a model.
    message = message.replace(" ", "")
    message = message.replace("​", "")
    message = message.replace("‍", " ")
    message = message.replace(" ", " ")
    message = message.replace("﻿", " ")
    message = message.replace("", "")
    message = message.replace("‎", "")

    # Use ftfy to fix any encoding issues.
    message = ftfy.fix_text(message, config=config)

    return message

def remove_excessive_newlines(message: str) -> str:
    return EXTRA_NEWLINE_PATTERN.sub("\n\n", message)

def remove_mentions(message: str) -> str:
    '''Attempts to remove any mentions (@name) from text.'''
    return MENTION_PATTERN.sub("", message).strip()

def remove_links(message: str) -> str:
    '''Removes any links from the given message, due to privacy concerns.'''
    no_links = LINKS_PATTERN.sub("", message).strip()
    # Make *sure* URLs are cleaned properly.
    assert "http://" not in no_links and "https://" not in no_links \
        , "Failed to clean URLs properly."
    
    return no_links

def remove_ooc(message: str) -> str:
    '''Attempts to remove any OOC from text.'''
    return OOC_PATTERN.sub("", message).strip()

def remove_trailing_whitespace_and_bad_lines(message: str) -> str:
    lines: list[str] = []
    for line in message.splitlines():
        # Trailing whitespace is always useless.
        line = line.rstrip()

        # Sometimes, users start their messages with "RE: (thread title, which
        # leaks usernames)" so we skip that here.
        if line.startswith("RE: ") or line.startswith("**RE: "):
            continue

        lines.append(line)

    return "\n".join(lines)
