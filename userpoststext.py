import time
from collections import Counter
from pathlib import Path

import numpy as np

from config import THIRD_PARTY_LIBRARIES_DIR, COMMON_LANGS, LANG_UNKNOWN, ALL_SENTIMENTS_RU, \
    ALL_SENTIMENTS_EN, ALL_SENTIMENTS_MUL, HUGGINGFACE_DIR, HUGGINGFACE_CACHE_DIR_OS_ENVIRONMENT_VAR, SLOW_MODE

import emoji
import fasttext

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import dostoevsky
from dostoevsky import data
from dostoevsky.models import FastTextSocialNetworkModel
from dostoevsky.models import FastTextToxicModel
from dostoevsky.tokenization import RegexTokenizer


import os
import re
import string

# language detection
LANG_MODEL = fasttext.load_model(str(THIRD_PARTY_LIBRARIES_DIR / 'lid.176.ftz'))

# sentiment analysis: RU
dostoevsky_model_path = THIRD_PARTY_LIBRARIES_DIR / 'fasttext-social-network-model.bin'
if not dostoevsky_model_path.is_file():
    model_name = 'fasttext-social-network-model.tar.xz'
    model_zip_path = THIRD_PARTY_LIBRARIES_DIR / model_name
    model_url = f'models/{model_name}'
    data_downloader = data.DataDownloader()
    data_downloader.download(source=model_url, destination=model_zip_path)
    model_zip_path.unlink(missing_ok=True)  # remove unnecessary zip file
FastTextSocialNetworkModel.MODEL_PATH = str(dostoevsky_model_path)
DOSTOEVSKY_SENTIMENT_MODEL = FastTextSocialNetworkModel(tokenizer=RegexTokenizer())


# sentiment analysis: EN
lexicon_file_path = THIRD_PARTY_LIBRARIES_DIR / 'sentiment' / 'vader_lexicon.zip'
assert lexicon_file_path.is_file()
nltk.data.path.append(THIRD_PARTY_LIBRARIES_DIR)
SIA = SentimentIntensityAnalyzer()

# sentiment analysis: MULTILINGUAL
os.environ[HUGGINGFACE_CACHE_DIR_OS_ENVIRONMENT_VAR] = str(HUGGINGFACE_DIR)  # strictly before import from transformers !
from transformers import pipeline  # strictly after making os environment TRANSFORMERS_CACHE !
MULTILINGUAL_SENTIMENT_MODEL_PATH = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
MULTILINGUAL_SENTIMENT_TASK = pipeline("sentiment-analysis",
                                       model=MULTILINGUAL_SENTIMENT_MODEL_PATH,
                                       tokenizer=MULTILINGUAL_SENTIMENT_MODEL_PATH)

last_time = time.time()


def print_with_time(s: str):
    current_time = time.time()
    global last_time
    print(f'{s}: {current_time - last_time:.2f} sec')
    last_time = current_time


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


def get_sentiments(users_texts: list[list[str]]) -> np.array:
    sentiments_ndarray_ = np.zeros((len(users_texts), len(ALL_SENTIMENTS_RU)))
    for user_num, user_texts_ in enumerate(users_texts):
        print_with_time(f'user_num: {user_num}')
        user_texts_ru = []
        user_texts_en = []
        user_texts_multilingual = []
        for t in user_texts_:
            if emoji.emoji_count(t) > 0:
                user_texts_multilingual.append(t)
                continue
            predict = LANG_MODEL.predict(t)[0][0][9:]
            if predict == 'ru':
                user_texts_ru.append(t)
            elif predict == 'en':
                user_texts_en.append(t)
            elif SLOW_MODE >= 2 and predict in ('ar', 'en', 'fr', 'de', 'hi', 'it', 'sp', 'pt'):
                user_texts_multilingual.append(t)
            else:
                continue

        if not user_texts_ru and not user_texts_en and not user_texts_multilingual:
            continue

        if user_texts_ru:
            try:
                user_results_ru = DOSTOEVSKY_SENTIMENT_MODEL.predict(user_texts_ru)
                for s_num, s_ in enumerate(ALL_SENTIMENTS_RU):
                    sentiments_ndarray_[user_num, s_num] += sum(r[s_] for r in user_results_ru)
            except RuntimeError as e:
                print(f'Sentiment predict RUSSIAN: user_num: {user_num}\n\n{e}\n\n')

        if user_texts_en:
            try:
                user_results_en = [SIA.polarity_scores(t) for t in user_texts_en]
                for s_num, s_ in enumerate(ALL_SENTIMENTS_EN):
                    sentiments_ndarray_[user_num, s_num] += sum(r[s_] for r in user_results_en)
            except RuntimeError as e:
                print(f'Sentiment predict ENGLISH: user_num: {user_num}\n\n{e}\n\n')

        if user_texts_multilingual:
            try:
                user_results_multilingual = MULTILINGUAL_SENTIMENT_TASK(user_texts_multilingual,
                                                                        return_all_scores=True)
                for s_num, s_ in enumerate(ALL_SENTIMENTS_MUL):
                    score = sum(d['score'] for l in user_results_multilingual for d in l if d['label'] == s_)
                    sentiments_ndarray_[user_num, s_num] += score
            except RuntimeError as e:
                print(f'Sentiment predict MULTILINGUAL: user_num: {user_num}\n\n{e}\n\n')

        for s_num in range(len(ALL_SENTIMENTS_RU)):
            sentiments_ndarray_[user_num, s_num] /= len(user_texts_)

        pass

    return sentiments_ndarray_

# def get_toxicity(text: str):
#     ???


if __name__ == '__main__':
    s = '🤔 🙈 😌 💕 👭 👙'
    print(extract_emojis(':) like! :( ;) ;( -p)'))
    print(extract_emojis(s))

    model = fasttext.load_model(str(THIRD_PARTY_LIBRARIES_DIR / 'lid.176.ftz'))
    # print(model.predict(['Ein, zwei, drei, vier' for _ in range(10000)], k=2))

    user_texts = ['Hello!', 'Special price! 30% off sales! Please call 0123456789',
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













