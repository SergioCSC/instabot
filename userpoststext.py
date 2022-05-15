from collections import Counter

import numpy as np

from nn_config import THIRD_PARTY_LIBRARIES_DIR, COMMON_LANGS, LANG_UNKNOWN, ALL_SENTIMENTS_RU, \
    ALL_SENTIMENTS_EN

import emoji
import fasttext

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import dostoevsky
from dostoevsky.models import FastTextSocialNetworkModel
from dostoevsky.models import FastTextToxicModel
from dostoevsky.tokenization import RegexTokenizer
from transformers import pipeline

import re
import string


FastTextSocialNetworkModel.MODEL_PATH = str(THIRD_PARTY_LIBRARIES_DIR / 'fasttext-social-network-model.bin')
DOSTOEVSKY_SENTIMENT_MODEL = FastTextSocialNetworkModel(tokenizer=RegexTokenizer())

LANG_MODEL = fasttext.load_model(str(THIRD_PARTY_LIBRARIES_DIR / 'lid.176.ftz'))

nltk.download('vader_lexicon')  # TODO if file not exist
SIA = SentimentIntensityAnalyzer()


def extract_emojis(s_: str):
    return ''.join(c for c in s_ if c in emoji.UNICODE_EMOJI['en'])


delimiters = string.punctuation + '\n\xa0«»\t—…' + '.,:; "\'`@&?!#()[]{}-+=*/|\\\t\n'
delimiters += ''.join(list(emoji.UNICODE_EMOJI['en']))
regexPattern = '|'.join(map(re.escape, set(delimiters)))
delimiters_regexp_compiled = re.compile(regexPattern)


def split_words(s_: str):
    return delimiters_regexp_compiled.split(s_)


def get_langs(users_texts: list[list[str]]) -> np.array:

    users_langs_ndarray_ = np.zeros((len(users_texts), len(COMMON_LANGS)))
    for user_num, user_texts_ in enumerate(users_texts):
        if not user_texts_:
            continue

        user_langs = Counter()

        predictions = LANG_MODEL.predict(user_texts_)
        for prediction in zip(predictions[0], predictions[1]):
            lang = prediction[0][0][9:]
            prob = prediction[1][0]
            if lang not in COMMON_LANGS:
                lang = LANG_UNKNOWN
            user_langs[lang] += prob

        for lang_num, lang in enumerate(COMMON_LANGS):
            users_langs_ndarray_[user_num, lang_num] = user_langs[lang] / len(user_texts_)

    return users_langs_ndarray_


def get_sentiments(users_texts: list[list[str]])  -> np.array:
    sentiments_ndarray_ = np.zeros((len(users_texts), len(ALL_SENTIMENTS_RU)))
    for user_num, user_texts_ in enumerate(users_texts):
        user_texts_ru = [t for t in user_texts_ if LANG_MODEL.predict(t)[0][0][9:] == 'ru']
        user_texts_en = [t for t in user_texts_ if LANG_MODEL.predict(t)[0][0][9:] == 'en']
        # TODO use languages besides english and russian
        if not user_texts_ru and not user_texts_en:
            continue
        if user_texts_ru:
            user_results_ru = DOSTOEVSKY_SENTIMENT_MODEL.predict(user_texts_ru)
            for s_num, s_ in enumerate(ALL_SENTIMENTS_RU):
                sentiments_ndarray_[user_num, s_num] += sum(r[s_] for r in user_results_ru) / len(user_texts_)
        if user_texts_en:
            user_results_en = [SIA.polarity_scores(t) for t in user_texts_en]
            for s_num, s_ in enumerate(ALL_SENTIMENTS_EN):
                sentiments_ndarray_[user_num, s_num] += sum(r[s_] for r in user_results_en) / len(user_texts_)
            pass
        pass

    return sentiments_ndarray_

# def get_toxicity(text: str):
#     ???


if __name__ == '__main__':
    s = '🤔 🙈 😌 💕 👭 👙'
    print(extract_emojis(':) like! :( ;) ;( -p)'))
    print(extract_emojis(s))

    model = fasttext.load_model(str(THIRD_PARTY_LIBRARIES_DIR / 'lid.176.ftz'))
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


    # # TODO support russian, not english only!
    # classifier = pipeline('zero-shot-classification')
    # users_businessness = classifier(user_texts, candidate_labels=['business'])

    FastTextSocialNetworkModel.MODEL_PATH = str(THIRD_PARTY_LIBRARIES_DIR / 'fasttext-social-network-model.bin')
    tokenizer = RegexTokenizer()
    model = FastTextSocialNetworkModel(tokenizer=tokenizer)
    results = model.predict(user_texts)
    pass













