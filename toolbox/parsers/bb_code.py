import logging
from re import search, sub

from transformers import pipeline

import bbcode


class BBCtoMD:
    IMAGE_FORMAT = ' An image of {}. '
    # Regex that matches if there are still tags, unformatted special characters, etc.
    INVALID_RESULT = r"<[^* '\n3A-Z=<\.]+>|\[[^A-Z* 0-9r\.]{0,10}\]|\\u|&[^ \n]{1,10};| |:[^ \n0-9]{1,10}:"
    IMAGE_UPDATE = "INSERT INTO img_cache (img_url, description, model) VALUES (?, ?, ?)"
    IMAGE_GET = "SELECT img_url, description FROM img_cache WHERE img_url = ? and model = ?"
    IMG_FAILURE_DESCRIPTIONS = set({
                                       'imgur': 'a sign that says "no parking"',
                                       'tinypic': 'a cartoon character with a cartoon face on a toy train'
                                   }.values())

    def __init__(self, img_model_name, cache_conn):
        # No reason to reinvent the wheel here
        self.parser = base_parser()
        self.youtube_count = 0
        self.links_count = 0
        # Inefficient/messy-looking, but keeping unicode
        # and special character mappings explicit, at least for now
        self.regex_replace = {
            # Emojis
            r":skull:|:dead:": "💀", r":thumb:|:thumbsup:": "👍",
            r":shock:|:astonished:": "😲", r":oops:": "🤭", r":undecided:": "😕",
            r":confused:": "😕", r":anxious:|:cold_sweat:": "😰", r":mad:|:angry:": "😠",
            r":furious:|:rage:": "😡", ":cry:": "😢",
            r"(\n){2,}": '\n',
            r"&#8230;|\\u2026|\u2026": '...',
            r"&quot;|\\u201d|\\u201c|\u201d|\u201c": '"',
            r"&copy;": '©', "&aring;": "å", "&alpha;": "α", "&ouml;": "ö",
            r"&amp;": '&', "&epsilon;": "ε", "&oacute;": "ó", "&hearts;": "♥",
            r"&#39;|\\u2019|\u2019|‘|\u2018|\\u2018|&#039;": '\'',
            r"\\u00e8": 'è',
            r"&lt;": "<",
            r"&gt;": ">",
            r"\\u00b0": '°',
            r"\\u2192": '→',
            r"\\u0101": 'ā',
            r"\\u3068": 'と',
            r"\\u3054": 'ご',
            r"\\u308f": 'わ',
            r"\\u305f": 'た',
            r"\\u3089": 'ら',
            r"\\u306a": 'な',
            r"\\u305c": 'ぜ',
            r"&mdash;|&ndash;|\\u2013|\u2013|\\u2014|\u2014": "--",
            r"\\u00e9|\u00e9": "é",
            r"\\u00ab|\u00ab": "«",
            # Remove any remaining links, nonsense user tags, unicode whitespaces, et cetera
            r"<a.+<\/a>|\[#pgcomm\]|-+\\n|-+\n|\\u012|\\u00a0|\[br\]| |<iframe.+>": '',
        }
        self.parse_image = image_parser(cache_conn, img_model_name)

        def user_tag(_tag_name, _value, opts, _parent, _context):
            # options = {
            #   'avatar': 'https://assets-cloud.enjin.com/users/3702764/avatar/small.1400073795.png',
            #   'name': 'Cleretic'}
            # value = '3702764'
            returning = opts['name'] + '.' if 'name' in opts else ''
            for option, opt_value in opts.items():
                if option == 'avatar':
                    description = self.parse_image(opt_value)
                    if len(description) > 0:
                        returning += self.IMAGE_FORMAT.format(description)
                elif option != 'name':
                    logging.warning('Unknown user option: %s', option)
            return returning

        self.parser.add_formatter('user', user_tag)

        def youtube_tag(_tag_name, _value, _opts, _parent, _context):
            # value '<a rel="nofollow" href="URL">NAME_OR_URL</a>'
            self.youtube_count += 1
            return ''

        self.parser.add_formatter('youtube', youtube_tag)

        def render_image(_tag_name, value, _options, _parent, _context):
            # value = '<img src="img_girl.jpg" alt="Girl in a jacket" width="500" height="600">'
            # options = { 'w': 420, 'h': 236}
            img_url = value.split('href="')[1].split('"')[0]
            description = self.parse_image(img_url)
            if len(description) > 0:
                return self.IMAGE_FORMAT.format(description)
            return description

        def render_link(_tag_name, value, _options, _parent, _context):
            # options = {'url': 'http://www.youtube.com/watch?v=zBIqLqUenz0'}
            # value = 'Gloomy Sunday'
            self.links_count += 1
            return value

        self.parser.add_formatter('a', render_link)
        self.parser.add_formatter('url', render_link)
        self.parser.add_formatter('img', render_image)
        # Users using brackets for things that should be italic
        self.bracket_regex = r'\[+(?:[^\]|]*\|)?([(low)|(sigh)|(open)|(closed)|(done)|(cont)|(eye)|(meant)^]*)\]+'

    def to_markdown(self, bbcode_str: str) -> str:
        for replace_str, with_str in self.regex_replace.items():
            bbcode_str = sub(replace_str, with_str, bbcode_str)
        formatted = self.parser.format(bbcode_str)
        for replace_str, with_str in self.regex_replace.items():
            formatted = sub(replace_str, with_str, formatted)
        formatted = sub(self.bracket_regex, r'*\1*', formatted)
        if search(self.INVALID_RESULT, formatted):
            logging.warning('Bad parsing.')
        return formatted


def base_parser():
    parser = bbcode.Parser(newline='\n', escape_html=False, replace_cosmetic=False)
    parse_as_remove = "%(value)s"
    bold = "**%(value)s**"
    parenthetical = f"({parse_as_remove})"
    parser.add_simple_formatter("b", bold)
    parser.add_simple_formatter("highlight", bold)
    parser.add_simple_formatter("size", bold)
    parser.add_simple_formatter("i", "*%(value)s*")
    parser.add_simple_formatter("u", "***%(value)s***")
    parser.add_simple_formatter("s", "(%(value)s)")
    parser.add_simple_formatter("hr", "<hr />", standalone=True)
    parentheticals = ['sub', 'sup', 'spoiler', 'warning']
    for paren in parentheticals:
        parser.add_simple_formatter(paren, parenthetical)
    parser.add_simple_formatter("table", "\n%(value)s\n")
    parser.add_simple_formatter("tr", "%(value)s\n")
    parser.add_simple_formatter("td", "%(value)s\t|")
    parser.add_simple_formatter("list", "\n%(value)s\n")
    parser.add_simple_formatter("*", "- %(value)s\n")
    parser.add_simple_formatter('email', '')  # Throw away emails
    # For these tags, simply discard them and keep the contents.
    for removing in ['rule', 'youtube', 'columns', 'span', 'div', 'font', 'justify', 'left',
                     'nextcol', 'txt', 'Raw Log', 'right', 'center', 'color', 'th',
                     'indent', 'red', 'blue', 'green', 'yellow', 'black', 'white']:
        parser.add_simple_formatter(removing, parse_as_remove)

    def render_quote(_tag_name, value, options, _parent, _context):
        quote = f"\"{value}\" "
        if 'name' in options:
            return f"{quote}-{options['name']}"
        return quote

    parser.add_formatter('quote', render_quote)
    return parser


def image_parser(image_cache_db, img_model_name):
    image_to_text = pipeline("image-to-text", model=img_model_name)

    def parse_image(img_url: str):
        for url, description_str in image_cache_db.execute(BBCtoMD.IMAGE_GET, (img_url, img_model_name)):
            return description_str
        try:
            description = image_to_text(img_url)
            description_str = ". ".join([desc['generated_text'].strip() for desc in description])
            if description_str in BBCtoMD.IMG_FAILURE_DESCRIPTIONS:
                description_str = ''
            image_cache_db.execute(BBCtoMD.IMAGE_UPDATE, (img_url, description_str, img_model_name))
            image_cache_db.commit()
            return description_str
        except Exception as ex:
            logging.error('Unidentified image, removing, cause: %s', ex)
            return ''

    return parse_image
