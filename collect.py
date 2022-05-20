import userpoststext
from nn_config import TRAIN_DATA_FILE, TEST_DATA_FILE, DATAFRAME_NAME, \
    FEATURES_DATA_FILE, FEATURES_NAME, DEPENDED_FEATURES_DATA_FILE, BOT_COL, \
    SAVED_PK, SAVED_UN, LEARNING_DATASETS_DIR, COMMON_LANGS, LANG_UNKNOWN, THIRD_PARTY_LIBRARIES_DIR, ALL_SENTIMENTS_RU

from csv2json import csv2json
import userpostsinfo as upi
from userpoststext import split_words, print_with_time, DOSTOEVSKY_SENTIMENT_MODEL
from zipf import estimate_zipf

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import emoji
# import torch
# from transformers import pipeline
# import fasttext


import dataclasses
import sys
import math
import time
import pickle
import statistics
from typing import Any
from pathlib import Path
from functools import cmp_to_key
from collections import defaultdict, Counter


EMPTY_VALUES_STR = {'', '0', '0.0', '0.00000', 'nan', 'none', 'None', 'UNKNOWN', '[]'}
UNIQUE_NUM_THRESHOLD = 10
NON_TRIVIAL_VALUES_FRACTION_THRESHOLD = 0.02


def most_popular_list_value(l_: list) -> Any:
    value_popularity_: defaultdict[str, tuple[int, Any]] = defaultdict(tuple[int, Any])
    for v_ in l_:
        count = value_popularity_[str(v_)][0] if str(v_) in value_popularity_ else 0
        value_popularity_[str(v_)] = (count + 1, v_)
    if not value_popularity_:
        return 0, 0, None
    sorted_value_popularity_: list[tuple[str, tuple[int, Any]]] \
        = sorted(list(value_popularity_.items()), key=lambda p: p[1][0], reverse=True)
    return (len(sorted_value_popularity_),) + sorted_value_popularity_[0][1]


def read_accounts_from_json_to_dataframe(filepath: Path) -> pd.DataFrame:

    if filepath:
        if filepath.suffix == '.csv':
            filepath = csv2json(filepath, -1)
        elif filepath.suffix == '.json':
            try:
                all_u = pd.read_json(filepath)
                if 'user' in all_u and 'status' in all_u and 'biography_with_entities' not in all_u:
                    filepath = csv2json(filepath, -1)
            except ValueError as e:
                filepath = csv2json(filepath, -1)
        else:
            raise NotImplementedError
        all_u = pd.read_json(filepath)

        feature_columns = pickle.load(open(FEATURES_DATA_FILE, 'rb'))

        for c in feature_columns:
            if c not in all_u.columns:
                all_u[c] = np.nan

        for c in all_u.columns:
            if c not in list(feature_columns):
                del all_u[c]

    else:
        # ACCOUNTS_JSONS_DIR = THIS_PYTHON_SCRIPT_DIR / 'parsed_users' / 'json_with_posts'
        # ACCOUNTS_JSONS_DIR = THIS_PYTHON_SCRIPT_DIR / 'parsed_users' / 'bots_detail_march'
        # all_u = pd.read_json(ACCOUNTS_JSONS_DIR / 'bots_1st_two.json')
        # all_u = pd.read_json(ACCOUNTS_JSONS_DIR / '50_bots.json')
        # all_u = all_u.append(pd.read_json(ACCOUNTS_JSONS_DIR / '327_bots.json'))
        # all_u = all_u.append(pd.read_json(ACCOUNTS_JSONS_DIR / '42_no_bots.json'))
        # all_u = all_u.append(pd.read_json(ACCOUNTS_JSONS_DIR / '_alexandra_arch_no_bots.json'))
        # all_u = all_u.append(pd.read_json(ACCOUNTS_JSONS_DIR / 'cha_food_no_bots.json'))
        # all_u = all_u.append(pd.read_json(ACCOUNTS_JSONS_DIR / 'alinkamoon_no_bots.json'))
        # all_u = all_u.append(pd.read_json(ACCOUNTS_JSONS_DIR / 'smagincartoonist_no_bots.json'))
        # all_u = pd.read_json(ACCOUNTS_JSONS_DIR / 'over_500_business_accounts.json')
        # all_u = pd.read_json(ACCOUNTS_JSONS_DIR / 'users_output_from_10_to_16_converted.json')
        # all_u = pd.read_json(ACCOUNTS_JSONS_DIR_7 / '9_different_users.json')
        # all_u = pd.read_json(ACCOUNTS_JSONS_DIR / '64_users_output_converted.json')
        # all_u = pd.read_json(ACCOUNTS_JSONS_DIR_2 / 'users_output_converted_marked.json')
        jsons_paths = LEARNING_DATASETS_DIR.glob('*.json')
        dataframes = (pd.read_json(path) for path in jsons_paths if path.is_file())
        all_u = pd.concat(dataframes)

        # all_u = all_u.sample(len(all_u) // 100)

        pickle.dump(all_u.columns, open(FEATURES_DATA_FILE, 'wb'))

    all_u.insert(0, SAVED_UN, all_u['username'])
    all_u.insert(0, SAVED_PK, all_u['pk'])
    print_with_time(f'read jsons. # accounts: {len(all_u)}')
    return all_u


def extract_depended_features(all_u: pd.DataFrame, inference_mode: bool) -> pd.DataFrame:
    def my_str_compare(s1: str, s2: str) -> int:
        if len(s1) != len(s2):
            return len(s1) - len(s2)
        return -1 if s1 < s2 else 0 if s1 == s2 else 1

    columns_to_keep = (BOT_COL, SAVED_PK, SAVED_UN)
    if not inference_mode:

        column_processing_info = {}
        for col in all_u:
            if col in columns_to_keep:
                continue
            _, _, most_popular_column_value = most_popular_list_value([v for v in all_u[col] if str(v) not in EMPTY_VALUES_STR])
            column_processing_info[col] = {'most_pop_0': type(most_popular_column_value)}
            if isinstance(most_popular_column_value, list):
                all_u[col] = [len(v) if isinstance(v, list) else 0 for v in all_u[col]]
            elif isinstance(most_popular_column_value, dict):
                all_u[col] = [str(v) for v in all_u[col]]

            unique_num, _, most_popular_column_value = most_popular_list_value([v for v in all_u[col] if str(v) not in EMPTY_VALUES_STR])
            column_processing_info[col]['most_pop_1'] = type(most_popular_column_value)
            if most_popular_column_value is None:
                del all_u[col]
                continue

            column_processing_info[col]['unique'] = unique_num
            if unique_num > UNIQUE_NUM_THRESHOLD or math.isnan(unique_num):
                if isinstance(most_popular_column_value, str):
                    all_u[col] = [0 if str(v) in EMPTY_VALUES_STR else 1 for v in all_u[col]]
                else:
                    all_u[col] = [0 if str(v) in EMPTY_VALUES_STR or isinstance(v, str) else v for v in all_u[col]]
            else:

                # get clases of values
                classes_of_values = set('' if str(v) in EMPTY_VALUES_STR else str(v) for v in all_u[col])
                column_processing_info[col]['classes'] = classes_of_values
                if len(classes_of_values) <= 1:
                    del all_u[col]
                    continue
                if classes_of_values == {'False', 'True'}:
                    all_u[col] = [0 if str(v) == 'False' else 1 for v in all_u[col]]
                    continue

                # sort classes of values by len(str(value))
                sorted_classes_of_values: list \
                    = sorted(list(classes_of_values), key=cmp_to_key(my_str_compare))

                # make classes from values
                all_u[col] = [0 if str(v) in EMPTY_VALUES_STR or str(v) not in classes_of_values
                              else sorted_classes_of_values.index(str(v)) + 1
                              for v in all_u[col]]

        print_with_time(f'cells: list, dict, str --> int')

        # filter out columns with very small fraction of non-trivial values
        for col in all_u:
            if col in columns_to_keep:
                continue  # don't filter such columns
            _, most_popular_popularity, _ = most_popular_list_value(list(all_u[col]))
            non_most_popular_values_fraction = 1 - most_popular_popularity / len(all_u)
            if non_most_popular_values_fraction < NON_TRIVIAL_VALUES_FRACTION_THRESHOLD:
                column_processing_info[col]['filter_out'] = 1
                del all_u[col]

        print_with_time(f'filter out columns with very small fraction of non-trivial values')

        # scale all columns to [0, 1]
        for col in all_u:
            if col in columns_to_keep:
                continue  # don't scale such columns
            left = min(all_u[col])
            right = max(all_u[col])
            column_processing_info[col]['scale_left'] = left
            column_processing_info[col]['scale_right'] = right

            all_u[col] = [(v - left) / (right - left) for v in all_u[col]]

            pass

        pickle.dump(column_processing_info, open(DEPENDED_FEATURES_DATA_FILE, 'wb'))

    else:
        column_processing_info = pickle.load(open(DEPENDED_FEATURES_DATA_FILE, 'rb'))

        for col in all_u:
            if col in columns_to_keep:
                continue
            most_popular_column_value_type = column_processing_info[col]['most_pop_0']
            if most_popular_column_value_type == list:
                all_u[col] = [len(v) if isinstance(v, list) else 0 for v in all_u[col]]
            elif most_popular_column_value_type == dict:
                all_u[col] = [str(v) for v in all_u[col]]

            most_popular_column_value_type = column_processing_info[col]['most_pop_1']
            if most_popular_column_value_type is type(None):
                del all_u[col]
                continue

            unique_num = column_processing_info[col]['unique']
            if unique_num > UNIQUE_NUM_THRESHOLD or math.isnan(unique_num):
                if most_popular_column_value_type == str:
                    all_u[col] = [0 if str(v) in EMPTY_VALUES_STR else 1 for v in all_u[col]]
                else:
                    all_u[col] = [0 if str(v) in EMPTY_VALUES_STR or isinstance(v, str) else v for v in all_u[col]]
            else:

                # get clases of values
                classes_of_values = column_processing_info[col]['classes']
                if len(classes_of_values) <= 1:
                    del all_u[col]
                    continue
                if classes_of_values == {'False', 'True'}:
                    all_u[col] = [0 if str(v) == 'False' else 1 for v in all_u[col]]
                    continue

                # sort classes of values by len(str(value))
                sorted_classes_of_values: list \
                    = sorted(list(classes_of_values), key=cmp_to_key(my_str_compare))
                # make classes from values
                all_u[col] = [0 if str(v) in EMPTY_VALUES_STR or str(v) not in classes_of_values
                              else sorted_classes_of_values.index(str(v)) + 1
                              for v in all_u[col]]

        print_with_time(f'cells: list, dict, str --> int')

        # filter out columns with very small fraction of non-trivial values
        for col in all_u:
            if col in columns_to_keep:
                continue  # don't filter out such columns
            if 'filter_out' in column_processing_info[col]:
                del all_u[col]

        print_with_time(f'filter out columns with very small fraction of non-trivial values')

        # scale all columns to [0, 1]
        for col in all_u:
            if col in columns_to_keep:
                continue  # don't scale such columns
            left = column_processing_info[col]['scale_left']
            right = column_processing_info[col]['scale_right']

            all_u[col] = [(v - left) / (right - left) for v in all_u[col]]

            pass
    print_with_time(f'scale all columns to [0, 1]')
    return all_u


def feature_extraction(all_u: pd.DataFrame, inference_mode: bool) -> pd.DataFrame:

    col = 'biography_with_entities'
    if col in all_u:
        all_u[col] = [len(v['entities']) if isinstance(v, dict) and 'entities' in v else 0 for v in all_u[col]]

    digits = '0123456789'
    all_u['digits_in_username'] = [len([c for c in str(v) if c in digits]) for v in all_u['username']]
    all_u['digits_in_biography'] = [len([c for c in str(v) if c in digits]) for v in all_u['biography']]
    all_u['сaps_in_full_name'] = [len([c for c in str(v) if c.lower() != c]) for v in all_u['full_name']]

    phone_in_biography = []
    for v in all_u['biography']:
        v = str(v)
        v = v.replace(' ', '').replace('-', '').replace('_', '').replace('(', '').replace(')', '')
        result = any(all(c in digits for c in v[i:i+5]) for i in range(len(v)-4))
        phone_in_biography.append(result)
    all_u['phone_in_biography'] = phone_in_biography

    POSTS_COLUMN = 'posts'

    users_posts = all_u[POSTS_COLUMN]  # [:10]
    users_texts = [[p['text'] for p in user_posts] for user_posts in users_posts]

    print_with_time('read user post texts')

    users_posts_lens = [[len(t) for t in user_texts] for user_texts in users_texts]
    users_total_lens = [sum(posts_lens) for posts_lens in users_posts_lens]
    users_average_post_lens = [sum(pl) / len(pl) if pl else 0 for pl in users_posts_lens]
    users_stdev_post_lens = [statistics.stdev(pl) if len(pl) > 1 else 0 for pl in users_posts_lens]

    all_u['total_posts_length'] = users_total_lens
    all_u['average_post_length'] = users_average_post_lens
    all_u['stdev_posts_length'] = users_stdev_post_lens
    #
    print_with_time('calc posts lengths: total, average, stdev')

    users_texts = [[t for t in user_texts if t] for user_texts in users_texts]
    users_texts = [[t if len(t) < 520 else t[:260] + t[-260:] for t in u] for u in users_texts]

    users_texts = [u if len(u) < 100 else sorted(u, key=len) for u in users_texts]
    users_texts = [u if len(u) < 100 else u[:25] + u[-75:] for u in users_texts]
    #
    # users_posts_emojis = [[[c for c in t if c in emoji.UNICODE_EMOJI['en']] for t in user_texts] for user_texts in users_texts]
    users_emoji_percents = [[len([c for c in t if c in emoji.UNICODE_EMOJI['en']])/len(t) for t in user_texts] for user_texts in users_texts]
    users_emoji_average_percent = [sum(user_emoji_percents)/len(user_emoji_percents) if user_emoji_percents else 0 for user_emoji_percents in users_emoji_percents]

    all_u['emoji_average_percent'] = users_emoji_average_percent

    print_with_time('extract emojies')

    users_vocabularies = [Counter(word for t in user_texts for word in split_words(t.lower())) for user_texts in users_texts]
    for user_vocabulary in users_vocabularies:
        del user_vocabulary['']
    users_word_counts = [np.array(sorted(v.values(), reverse=True)) for v in users_vocabularies]
    users_alpha, users_c = zip(*[estimate_zipf(wc) for wc in users_word_counts])
    all_u['users_alpha'] = users_alpha
    all_u['users_c'] = users_c
    print_with_time('extract users vocabularies')

    users_texts = [[t for t in user_texts if len(t) > 5] for user_texts in users_texts]  # TODO const

    sentiments_ndarray_ = userpoststext.get_sentiments(users_texts)
    columns = [f'mood_{s}' for s in ALL_SENTIMENTS_RU]
    all_u = pd.concat([all_u, pd.DataFrame(sentiments_ndarray_, index=all_u.index, columns=columns)], axis=1)

    print_with_time('extract sentiments')

    users_texts = [[''.join(c for c in t if c not in emoji.UNICODE_EMOJI['en']) for t in user_texts] for user_texts in users_texts]

    users_langs_ndarray = userpoststext.get_langs(users_texts)

    columns = [f'lang_{lang}' for lang in COMMON_LANGS]
    all_u = pd.concat([all_u, pd.DataFrame(users_langs_ndarray, index=all_u.index, columns=columns)], axis=1)

    print_with_time('extract langs')

    users_texts = [[t for t in user_texts if len(t.split()) > 1] for user_texts in users_texts]  # TODO const


    # # TODO support russian, not english only!
    # classifier = pipeline('text-classification', model='mrm8488/bert-tiny-finetuned-sms-spam-detection')
    # users_spams_dicts = [classifier([t for t in user_texts if isinstance(t, str) and len(t) > 10]) if user_texts else () for user_texts in users_texts]
    # users_spams = [[spam_dict['score'] if spam_dict['label'] == 'LABEL_0' else 1 - spam_dict['score'] for spam_dict in user_spam_dicts] for user_spam_dicts in users_spams_dicts]
    # all_u['users_spams'] = users_spams
    #
    # print_with_time('calculate which messages could be spams')

    # too slow
    # # TODO support russian, not english only!
    # classifier = pipeline('zero-shot-classification')
    # is_business = lambda texts: [r['scores'][0] for r in classifier(texts, candidate_labels=['business'])]
    # users_businessness = [is_business(user_texts) for user_texts in users_texts]
    # all_u['users_businessness'] = users_businessness
    #
    # print_with_time('calculate which messages could be business')

    all_u[POSTS_COLUMN] = [v if not (isinstance(v, float) and math.isnan(v)) else [] for v in all_u[POSTS_COLUMN]]

    all_u[upi.POSTS_N] = [len(v) for v in all_u[POSTS_COLUMN]]
    all_u[upi.LIKES_N] = [sum([int(p['likes_count']) for p in v]) for v in all_u[POSTS_COLUMN]]
    all_u[upi.COMMENTS_N] = [sum([int(p['comments_count']) for p in v]) for v in all_u[POSTS_COLUMN]]

    print_with_time(f'calc posts likes comments')

    users_posts_info: list[upi.UserPostsInfo] = []
    for i in range(len(all_u)):
        user_posts = all_u[POSTS_COLUMN].iloc[i]
        user_posts_info = upi.UserPostsInfo()

        for post in user_posts:
            likes_count = int(post['likes_count'])  # post_parts[-3])
            comments_count = int(post['comments_count'])
            date_ = post['date']
            time_ = post['time']
            hours, minutes = (int(t) for t in time_.split(':'))
            try:
                upi.pick_by_minutes(user_posts_info, minutes, 1, likes_count, comments_count)
                upi.pick_by_hours(user_posts_info, hours, 1, likes_count, comments_count)
                upi.pick_by_date(user_posts_info, date_, 1, likes_count, comments_count)
            except ValueError as e:
                # print(f'Error date in post: {post["date"]}'
                #       f' user pk: {all_u["pk"].iloc[i]} username: {all_u["username"].iloc[i]}'
                #       f' post_id: {post["post_id"]}'
                #       f' post_url: {post["post_url"]}')
                # print('!', end='')
                pass
            upi.add_to_counter(user_posts_info.overall, 1, likes_count, comments_count)

        users_posts_info.append(user_posts_info)
        pass

    print_with_time(f'fill list of UserPostInfo (=set of counters)')  # 6 sec with @dataclass of Counter's filling

    # calibrate: v --> v / overall_v for all post, likes, comments values
    for user_posts_info in users_posts_info:
        overall_value = getattr(user_posts_info, 'overall')
        for attr in user_posts_info.__annotations__:
            if attr != 'overall':
                attr_value = getattr(user_posts_info, attr)
                for k in overall_value:
                    attr_value[k] = attr_value[k] / overall_value[k] if attr_value[k] else 0.0

    print_with_time(f'calibrate: v --> v / overall_v for all post, likes, comments values')

    # calc variances
    upi.fill_variances(users_posts_info)

    print_with_time(f'calc variances')

    post_columns = [f'{c}_{s}' for s in upi.UserPostsInfo.__annotations__ for c in (upi.POSTS_N, upi.LIKES_N, upi.COMMENTS_N)]
    # all_u_extra = pd.DataFrame(0.0 838 строк, columns=columns)
    # all_u[post_columns] = np.nan  # performance warning
    # all_u = all_u.copy()

    users_post_dataframes = []
    users_post_ndarray_ = np.zeros((len(all_u), len(post_columns)))

    for i, user_posts_info in enumerate(users_posts_info):

        user_posts_dataframe = upi.make_user_posts_dataframe()
        j = 0
        for k in upi.UserPostsInfo.__annotations__:
            posts_likes_comments_num = user_posts_info.__getattribute__(k)
            # user_posts_dataframe.loc[k] = posts_likes_comments_num[upi.POSTS_N], \
            #                               posts_likes_comments_num[upi.LIKES_N], \
            #                               posts_likes_comments_num[upi.COMMENTS_N]

            # all_u[f'{upi.POSTS_N}_{k}'].iloc[i] = posts_likes_comments_num[upi.POSTS_N]
            # all_u[f'{upi.LIKES_N}_{k}'].iloc[i] = posts_likes_comments_num[upi.LIKES_N]
            # all_u[f'{upi.COMMENTS_N}_{k}'].iloc[i] = posts_likes_comments_num[upi.COMMENTS_N]

            users_post_ndarray_[i, j] = posts_likes_comments_num[upi.POSTS_N]
            users_post_ndarray_[i, j + 1] = posts_likes_comments_num[upi.LIKES_N]
            users_post_ndarray_[i, j + 2] = posts_likes_comments_num[upi.COMMENTS_N]
            j += 3

        users_post_dataframes.append(user_posts_dataframe)

    print_with_time(f'ndarray from counters')  # 4 minutes with Dataframes filling

    # make all_u columns from posts likes comments rates
    all_u = pd.concat([all_u, pd.DataFrame(users_post_ndarray_, index=all_u.index, columns=post_columns)], axis=1)
    # all_u = all_u.copy()  # defragmentation of dataframe

    print_with_time(f'add ndarray to all_u')

    all_u = extract_depended_features(all_u, inference_mode)
    return all_u.copy()


def feature_selection(all_u: pd.DataFrame) -> pd.DataFrame:

    # corr = all_u.corr(method='pearson')
    # shuffled_all_u = all_u.sample(frac=1)
    # auto_correlations = sorted([(c, (shuffled_all_u[c]).autocorr()) for c in all_u], key=lambda p: p[1])

    del all_u['external_lynx_url']  # correlation with 'external_url' == 1
    del all_u['can_hide_public_contacts']  # correlation with 'can_hide_category' == 1
    del all_u['charity_profile_fundraiser_info']  # correlation with 'is_business' == 1
    # del all_u['contact_phone_number']  # correlation with 'public_phone_number' == 1
    # del all_u['city_name']  # correlation with 'city_id' == 0.9998

    # del all_u['is_eligible_for_smb_support_flow']  # correlation with 'is_interest_account' == 0.97
    del all_u['posts']  # same as 'p_overall'
    # del all_u['city_id']  # correlation with latitude > 0.97
    # del all_u['public_phone_country_code']  # corr with public_phone_number > 0.97
    # corr = all_u.corr(method='pearson')

    # del all_u['pk']  # correlation with BOT_COL == 0.92, look like cheat
    # del all_u['interop_messaging_user_fbid']  # correlation with BOT_COL == 0.93, look like cheat
    del all_u['l_autumn_r']  # corr with p_autumn_r > 0.98
    del all_u['l_spring_r']  # corr with p_spring_r > 0.98
    del all_u['l_winter_r']  # corr with p_winter_r > 0.98
    del all_u['l_d_var']  # corr with p_d_var > 0.97
    del all_u['l_d_01_to_10_r']  # corr with p_d_01_to_10_r > 0.99
    del all_u['l_d_11_to_20_r']  # corr with p_d_11_to_20_r > 0.98
    del all_u['l_h_21_to_3_r']  # corr with p_h_21_to_3_r > 0.98
    del all_u['l_h_15_to_21_r']  # corr with p_h_15_to_21_r > 0.98
    del all_u['l_h_9_to_15_r']  # corr with p_h_9_to_15_r > 0.98

    # low importance due to 1 neuron learning
    # del all_u['full_name']
    # del all_u['follower_count']
    # del all_u['usertags_count']
    # del all_u['hd_profile_pic_versions']
    # del all_u['longitude']
    # del all_u['public_phone_country_code']

    print_with_time(f'del highly correlated columns')

    # corr = all_u.corr(method='pearson')
    # print(min(corr))

    return all_u.copy()


def check_test_columns_matches_train_columns(test_dataframe: pd.DataFrame):
    try:
        train_store = pd.HDFStore(TRAIN_DATA_FILE, mode='r')
    except OSError as e:
        print(f'no train dataframe file: {TRAIN_DATA_FILE}')
        return
    train_dataframe = train_store[DATAFRAME_NAME]
    train_store.close()
    assert set(train_dataframe.columns) == set(test_dataframe.columns) - set((SAVED_PK, SAVED_UN))


def save_features_as_train_and_test(all_u: pd.DataFrame, test_size):

    if test_size == 1:
        test_store = pd.HDFStore(TEST_DATA_FILE, mode='w')
        test_store[DATAFRAME_NAME] = all_u
        test_store.close()
    else:
        X_train, _X_test = train_test_split(
            all_u,
            test_size=test_size,
            shuffle=True)

        del X_train[SAVED_PK]
        del X_train[SAVED_UN]

        train_store = pd.HDFStore(TRAIN_DATA_FILE, mode='w')
        test_store = pd.HDFStore(TEST_DATA_FILE, mode='w')

        train_store[DATAFRAME_NAME] = X_train
        test_store[DATAFRAME_NAME] = _X_test

        train_store.close()
        test_store.close()

    print_with_time('\nstore train and test dataframes in files')


def collect_and_save_features(inference_accounts_filepath_: Path):

    start_time = time.time()
    all_users_dataframe: pd.DataFrame = read_accounts_from_json_to_dataframe(inference_accounts_filepath_)
    print(f'all_users_dataframe.shape: {all_users_dataframe.shape}')
    all_users_dataframe = feature_extraction(all_users_dataframe, bool(inference_accounts_filepath_))
    print(f'all_users_dataframe.shape: {all_users_dataframe.shape}')
    all_users_dataframe = feature_selection(all_users_dataframe)
    print(f'all_users_dataframe.shape: {all_users_dataframe.shape}')

    if inference_accounts_filepath_:
        check_test_columns_matches_train_columns(all_users_dataframe)

    save_features_as_train_and_test(all_users_dataframe, 1.0 if inference_accounts_filepath_ else 0.2)
    print(f'total time: {time.time() - start_time} sec')


if __name__ == '__main__':
    inference_accounts_filepath = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    collect_and_save_features(inference_accounts_filepath)
