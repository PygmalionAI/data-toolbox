import logging

from transformers import pipeline

import bbcode


class BBCtoMD:

    def __init__(self, img_model_name):
        # No reason to reinvent the wheel here
        # todo verify all of this; no good yet
        self.parser = bbcode.Parser(newline='\n', escape_html=False, replace_cosmetic=False)
        self.parse_as_remove = "%(value)s"
        self.parser.add_simple_formatter("b", "**%(value)s**")
        self.parser.add_simple_formatter("size", "**%(value)s**")
        self.parser.add_simple_formatter("i", "*%(value)s*")
        self.parser.add_simple_formatter("u", "***%(value)s***")
        self.parser.add_simple_formatter("s", "<strike>%(value)s</strike>")
        self.parser.add_simple_formatter("hr", "<hr />", standalone=True)
        self.parser.add_simple_formatter("sub", "<sub>%(value)s</sub>")
        self.parser.add_simple_formatter("sup", "<sup>%(value)s</sup>")

        # For these tags, simply discard them and keep the contents.
        for removing in ['rule', 'youtube', 'table', 'tr', 'td', 'columns', 'span', 'div', 'spoiler',
                         'indent', 'red', 'blue', 'green', 'yellow', 'black', 'white']:
            self.parser.add_simple_formatter(removing, self.parse_as_remove)

        self.img_count = 0
        self.errored_images = 0
        self.replacements = {
            "&quot;": "\"",
            '&#8230;': "...",
            '&#39;': '\'',
            '&amp;': '&',
            '\n\n': '\n',
            "&copy;": '©',
            "&lt;": "<",
            "&gt;": ">"
        }
        self.image_to_text = pipeline("image-to-text", model=img_model_name)
        self.imgur_failure_description = 'a sign that says "no parking"'

        def render_image(_tag_name, value, _options, _parent, _context):
            # value = '<img src="img_girl.jpg" alt="Girl in a jacket" width="500" height="600">'
            # options = { 'w': 420, 'h': 236}
            self.img_count += 1
            img_url = value.split('href="')[1].split('"')[0]
            try:
                description = self.image_to_text(img_url)
                description_str = ". ".join([desc['generated_text'].strip() for desc in description])
                if description_str == self.imgur_failure_description:
                    return ''
                return f'<img alt="{description_str}">'
            except Exception as ex:
                logging.error('Unidentified image, removing, cause: %s', ex)
                self.errored_images += 1
                return ''

        def render_link(_tag_name, value, _options, _parent, _context):
            # options = {'url': 'http://www.youtube.com/watch?v=zBIqLqUenz0'}
            # value = 'Gloomy Sunday'
            return value

        self.parser.add_formatter('a', render_link)
        self.parser.add_formatter('url', render_link)
        self.parser.add_formatter('img', render_image)

    def to_markdown(self, bbcode_str: str) -> str:
        formatted = self.parser.format(bbcode_str)
        for replace_str, with_str in self.replacements.items():
            formatted = formatted.replace(replace_str, with_str)
        return formatted
