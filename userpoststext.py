import re
import string

import emoji
import fasttext

from transformers import pipeline


def extract_emojis(s_: str):
    return ''.join(c for c in s_ if c in emoji.UNICODE_EMOJI['en'])


delimiters = string.punctuation + '\n\xa0«»\t—…' + '.,:; "\'`@&?!#()[]{}-+=*/|\\\t\n'
delimiters += ''.join(list(emoji.UNICODE_EMOJI['en']))
regexPattern = '|'.join(map(re.escape, set(delimiters)))
delimiters_regexp_compiled = re.compile(regexPattern)


def split_words(s_: str):
    return delimiters_regexp_compiled.split(s_)


if __name__ == '__main__':
    s = '🤔 🙈 😌 💕 👭 👙'
    print(extract_emojis(':) like! :( ;) ;( -p)'))
    print(extract_emojis(s))

    model = fasttext.load_model('lid.176.ftz')
    print(model.predict('الشمس تشرق')[0][0])
    print(model.predict('الشمس تشرق', k=2))
    print(model.predict('影響包含對氣候的變化以及自然資源的枯竭程度', k=2))
    # print(model.predict(['Ein, zwei, drei, vier' for _ in range(10000)], k=2))

    user_texts = ['Hello!', 'Special price! 30% off sales! Please call 89728658235',
                  'Ах ты мой милый', 'Папа у Васи силён. Но он слишком умный, сцуко']

    # classifier = pipeline('sentiment-analysis')
    # classifier = pipeline('sentiment-analysis', model='blanchefort/rubert-base-cased-sentiment')
    # results = classifier(user_texts)
    # results = classifier(user_texts)
    # print(results)
    # classifier = pipeline("text-classification", model="SkolkovoInstitute/russian_toxicity_classifier")
    # results = classifier(user_texts)
    # # print(results)
    # classifier = pipeline('text-classification', model='mrm8488/bert-tiny-finetuned-sms-spam-detection')
    # results = classifier(user_texts, return_all_scores=True)
    # print(results)


    # TODO support russian, not english only!
    classifier = pipeline('zero-shot-classification')
    users_businessness = classifier(user_texts, candidate_labels=['business'])
    pass













