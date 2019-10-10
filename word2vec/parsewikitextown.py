import re

from gensim import utils
from gensim.corpora.wikicorpus import remove_template, remove_file, \
    RE_P0, RE_P1, RE_P2, RE_P9, RE_P10, RE_P11, RE_P14, RE_P5, RE_P12, RE_P13

#https://www.mediawiki.org/wiki/Help:Formatting

RE_P2a = re.compile(r'(\n\[\[[a-z][a-z][\w-]*:[^:\]]+\]\])+$', re.MULTILINE | re.UNICODE)

RE_P20 = re.compile(r'\[\[(.*?)(\|.*?)?\]\]([a-z\']*)', re.DOTALL | re.UNICODE)
"""links with optional trail"""
#re.compile(r'\[?\[(.*?)(\|.*?)*?\]\]?', re.DOTALL | re.UNICODE) # with file and image links

RE_P21 = re.compile(r'\'{2,}(.*?)\'{2,}', re.DOTALL | re.UNICODE)
"""italic and bold"""

RE_P22 = re.compile(r'={2,}(.*?)={2,}', re.DOTALL | re.UNICODE)
"""headings"""


class MarkupLinks(object):
    def __init__(self):
        self.links = dict()

    def process_text(self, text):
        return text.strip().lower()

    def __make_link_url(self, link_text):
        return ' http://dbpedia.org/resource/' + link_text.replace(' ', '_')

    def __call__(self, match):
        link = match.group(1)
        text = match.group(2)
        trail = match.group(3)
        if text:
            if len(text) > 1:
                text = text[1:]
        else:
            text = link


        if link == 'of' or link == 'dor':
            print("test")
        if text == 'of' or text == 'dor':
            print("test")

        self.links[link] = link #add link to map
        self.links[text] = link  # add link text to map
        if trail:
            self.links[text + trail] = link  # add link text to map
            return text + trail
        return text

def __remove_markup(text):
    links = MarkupLinks()

    text = re.sub(RE_P2a, '', text)  # remove the last list (=languages)
    # the wiki markup is recursive (markup inside markup etc)
    # instead of writing a recursive grammar, here we deal with that by removing
    # markup in a loop, starting with inner-most expressions and working outwards,
    # for as long as something changes.
    text = remove_template(text)
    text = remove_file(text)
    iters = 0
    while True:
        old, iters = text, iters + 1# remove comments
        text = re.sub(RE_P1, '', text)  # remove footnotes
        text = re.sub(RE_P9, '', text)  # remove outside links
        text = re.sub(RE_P10, '', text)  # remove math content
        text = re.sub(RE_P11, '', text)  # remove all remaining tags
        text = re.sub(RE_P14, '', text)  # remove categories
        text = re.sub(RE_P5, '\\3', text)  # remove urls, keep description
        text = re.sub(RE_P0, '', text)
        text = re.sub(RE_P21, '\\1', text)  # italic and bold
        text = re.sub(RE_P22, '\\1', text)  # headline
        text = re.sub(RE_P20, links, text)  # links

        # remove table markup
        text = text.replace('||', '\n|')  # each table cell on a separate line
        text = re.sub(RE_P12, '\n', text)  # remove formatting lines
        text = re.sub(RE_P13, '\n\\3', text)  # leave only cell content
        # remove empty mark-up
        text = text.replace('[]', '')
        # stop if nothing changed between two iterations or after a fixed number of iterations
        if old == text or iters > 2:
            break

    text = text.replace('[', '').replace(']', '')  # promote all remaining markup to plain text
    text = text.replace('*', '').replace('#', '').replace(';', '').replace(':', '')

    return text, links.links


def get_raw_text_and_links_from_markup(raw):
    if raw == None:
        raw = ''
    text = utils.to_unicode(raw, 'utf-8', errors='ignore')
    text = utils.decode_htmlentities(text)  # '&amp;nbsp;' --> '\xa0'
    return __remove_markup(text)
