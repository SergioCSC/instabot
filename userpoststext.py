import re
import string

import emoji
import fasttext


def extract_emojis(s_: str):
    return ''.join(c for c in s_ if c in emoji.UNICODE_EMOJI['en'])


delimiters = string.punctuation + '\n\xa0«»\t—…' + '.,:; "\'`@&?!#()[]{}-+=*/|\\\t\n'
delimiters += ''.join(list(emoji.UNICODE_EMOJI['en']))
regexPattern = '|'.join(map(re.escape, set(delimiters)))
delimiters_regexp_compiled = re.compile(regexPattern)


def split_words(s_: str):
    return delimiters_regexp_compiled.split(s_)


s = '🤔 🙈 😌 💕 👭 👙'
print(extract_emojis(':) like! :( ;) ;( -p)'))
print(extract_emojis(s))

model = fasttext.load_model('lid.176.ftz')
print(model.predict('الشمس تشرق')[0][0])
print(model.predict('الشمس تشرق', k=2))
print(model.predict('影響包含對氣候的變化以及自然資源的枯竭程度', k=2))
# print(model.predict(['Ein, zwei, drei, vier' for _ in range(10000)], k=2))
















