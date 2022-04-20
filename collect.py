from nn_config import TRAIN_DATA_FILE, TEST_DATA_FILE, DATAFRAME_NAME
from userpoststext import split_words
from zipf import estimate_zipf
import userpostsinfo as upi

import statistics
from transformers import pipeline

import emoji
import numpy as np
import fasttext


from collections import defaultdict, Counter
from pathlib import Path
from typing import Any

import pandas as pd
import math
import time
import dataclasses

from sklearn.model_selection import train_test_split
import torch


EMPTY_VALUES_STR = {'', '0', '0.0', '0.00000', 'nan', 'none', 'None', 'UNKNOWN', '[]'}
UNIQUE_NUM_THRESHOLD = 10
NON_TRIVIAL_VALUES_FRACTION_THRESHOLD = 0.02
last_time = time.time()


def print_with_time(s: str):
    current_time = time.time()
    global last_time
    print(f'{s}: {current_time - last_time:.2f} sec')
    last_time = current_time


def most_popular_list_value(l_: list) -> Any:
    value_popularity_: defaultdict[str, tuple[int, Any]] = defaultdict(tuple[int, Any])
    for v_ in l_:
        count = value_popularity_[str(v_)][0] if str(v_) in value_popularity_ else 0
        value_popularity_[str(v_)] = (count + 1, v_)
    if not value_popularity_:
        return 0, None
    sorted_value_popularity_: list[tuple[str, tuple[int, Any]]] \
        = sorted(list(value_popularity_.items()), key=lambda p: p[1][0], reverse=True)
    return sorted_value_popularity_[0][1]


def read_accounts_from_json_to_dataframe() -> pd.DataFrame:

    THIS_PYTHON_SCRIPT_DIR = Path(__file__).resolve().parent
    USERS_DIR = THIS_PYTHON_SCRIPT_DIR / 'parsed_users' / 'bots_business_april' / 'users'
    # USERS_DIR = THIS_PYTHON_SCRIPT_DIR / 'parsed_users' / 'json_with_posts'
    # USERS_DIR = THIS_PYTHON_SCRIPT_DIR / 'parsed_users' / 'bots_detail_march'
    # all_u = pd.read_json(USERS_DIR / 'bots_1st_two.json')
    # all_u = pd.read_json(USERS_DIR / '50_bots.json')
    # all_u = all_u.append(pd.read_json(USERS_DIR / '327_bots.json'))
    # all_u = all_u.append(pd.read_json(USERS_DIR / '42_no_bots.json'))
    # all_u = all_u.append(pd.read_json(USERS_DIR / '_alexandra_arch_no_bots.json'))
    # all_u = all_u.append(pd.read_json(USERS_DIR / 'cha_food_no_bots.json'))
    # all_u = all_u.append(pd.read_json(USERS_DIR / 'alinkamoon_no_bots.json'))
    # all_u = all_u.append(pd.read_json(USERS_DIR / 'smagincartoonist_no_bots.json'))
    # all_u = pd.read_json(USERS_DIR / 'over_500_business_accounts.json')
    all_u = pd.read_json(USERS_DIR / 'users_output_from_10_to_16_converted.json')
    # all_u = pd.read_json(USERS_DIR / '32_users_output_converted.json')

    print_with_time('read jsons')
    return all_u


def feature_extraction(all_u: pd.DataFrame):

    col = 'biography_with_entities'
    all_u[col] = [len(v['entities']) for v in all_u[col]]

    POSTS_COLUMN = 'posts'

    users_posts = all_u[POSTS_COLUMN]  # [:10]
    users_texts = [[p['text'] for p in user_posts] for user_posts in users_posts]

    print_with_time('read user post texts')

    users_posts_lens = [[len(t) for t in user_texts] for user_texts in users_texts]
    users_total_lens = [sum(posts_lens) for posts_lens in users_posts_lens]
    users_average_post_lens = [sum(pl) / len(pl) if pl else 0 for pl in users_posts_lens]
    users_stdev_post_lens = [statistics.stdev(pl) if len(pl) > 1 else 0 for pl in users_posts_lens]

    print_with_time('calc posts lengths: total, average, stdev')

    users_posts_emojis = [[[c for c in t if c in emoji.UNICODE_EMOJI['en']] for t in user_texts] for user_texts in users_texts]
    users_emoji_percents = [[len([c for c in t if c in emoji.UNICODE_EMOJI['en']])/len(t) for t in user_texts] for user_texts in users_texts]
    users_emoji_average_percent = [sum(user_emoji_percents)/len(user_emoji_percents) if user_emoji_percents else 0 for user_emoji_percents in users_emoji_percents]

    print_with_time('extract emojies')

    model = fasttext.load_model('lid.176.ftz')
    users_posts_langs = [Counter(model.predict(t)[0][0][9:] for t in user_texts) for user_texts in users_texts]

    print_with_time('extract langs')

    users_vocabularies = [Counter(word for t in user_texts for word in split_words(t.lower())) for user_texts in users_texts]
    for user_vocabulary in users_vocabularies:
        del user_vocabulary['']
    users_word_counts = [np.array(sorted(v.values(), reverse=True)) for v in users_vocabularies]
    users_alpha_and_C = [estimate_zipf(wc) for wc in users_word_counts]

    print_with_time('extract users vocabularies')

    # TODO support russian, not english only!
    # classifier = pipeline('text-classification', model='mrm8488/bert-tiny-finetuned-sms-spam-detection')
    # users_spams_dicts = [classifier(user_texts) for user_texts in users_texts]
    # users_spams = [[spam['score'] if spam['label'] == 'LABEL_0' else 1 - spam['score'] for spam in user_spams] for user_spams in users_spams_dicts]

    print_with_time('calculate which messages could be spams')

    # TODO support russian, not english only!
    # classifier = pipeline('zero-shot-classification')
    # is_business = lambda texts: [r['scores'][0] for r in classifier(texts, candidate_labels=['business'])]
    # users_businessness = [is_business(user_texts) for user_texts in users_texts]

    print_with_time('calculate which messages could be business')
#
    all_u['total_posts_length'] = users_total_lens
    all_u['average_post_length'] = users_average_post_lens
    all_u['stdev_posts_length'] = users_stdev_post_lens

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
                raise ValueError(f'user: {i} post: {post}') from e
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

    i = 0
    for user_posts_info in users_posts_info:

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
        i += 1

    print_with_time(f'ndarray from counters')  # 4 minutes with Dataframes filling

    # make all_u columns from posts likes comments rates
    # all_u[post_columns] = pd.DataFrame(users_post_ndarray_)  # bug: rows with same label
    # all_u = pd.concat([all_u, pd.DataFrame(users_post_ndarray_)], axis=1)
    # all_u.join(pd.DataFrame(users_post_ndarray_))
    # all_u[post_columns[0]] = users_post_ndarray_[:, 0]
    all_u[post_columns] = pd.DataFrame(users_post_ndarray_, index=all_u.index)  # bug: rows with same label
    all_u = all_u.copy()  # defragmentation of dataframe

    print_with_time(f'add ndarray to all_u')

    description = all_u.describe(include='all')  # .loc['unique', :]

    print_with_time(f'all_u.descibe()')

    column_i = 0
    for col in all_u:
        _, most_popular_column_value = most_popular_list_value([v for v in all_u[col] if str(v) not in EMPTY_VALUES_STR])
        if isinstance(most_popular_column_value, list):
            all_u[col] = [len(v) if isinstance(v, list) else 0 for v in all_u[col]]
        elif isinstance(most_popular_column_value, dict):
            all_u[col] = [str(v) for v in all_u[col]]

        _, most_popular_column_value = most_popular_list_value([v for v in all_u[col] if str(v) not in EMPTY_VALUES_STR])
        if most_popular_column_value is None:
            del all_u[col]
            continue

        unique_num = description[col]['unique']
        if unique_num > UNIQUE_NUM_THRESHOLD or math.isnan(unique_num):
            if isinstance(most_popular_column_value, str):
                all_u[col] = [0 if str(v) in EMPTY_VALUES_STR else 1 for v in all_u[col]]
            else:
                all_u[col] = [0 if str(v) in EMPTY_VALUES_STR or isinstance(v, str) else v for v in all_u[col]]
        else:
            # if isinstance(most_popular_column_value, bool):
            #     # all_u[col] = [int(v) if isinstance(v, bool) else for v in all_u[col]]
            #     # continue
            #     pass

            # get clases of values
            classes_of_values = set('' if str(v) in EMPTY_VALUES_STR else str(v) for v in all_u[col])
            if len(classes_of_values) <= 1:
                del all_u[col]
                continue
            if classes_of_values == {'False', 'True'}:
                all_u[col] = [0 if str(v) == 'False' else 1 for v in all_u[col]]
                continue

            # sort classes of values by len(str(value))
            sorted_classes_of_values: list \
                = sorted(list(classes_of_values), key=lambda p: len(p))

            # make classes from values
            all_u[col] = [0 if str(v) in EMPTY_VALUES_STR else sorted_classes_of_values.index(str(v)) for v in all_u[col]]

        column_i += 1


    print_with_time(f'cells: list, dict, str --> int')

    # filter out columns with very small fraction of non-trivial values
    for col in all_u:
        if col == 'bot':
            continue  # don't filter 'bot' column
        most_popular_popularity, _ = most_popular_list_value(list(all_u[col]))
        non_most_popular_values_fraction = 1 - most_popular_popularity / len(all_u)
        if non_most_popular_values_fraction < NON_TRIVIAL_VALUES_FRACTION_THRESHOLD:
            del all_u[col]

    print_with_time(f'filter out columns with very small fraction of non-trivial values')

    # scale all columns to [0, 1]
    for col in all_u:
        if col == 'bot':
            continue  # don't scale 'bot' column
        left = min(all_u[col])
        right = max(all_u[col])
        all_u[col] = [(v - left) / (right - left) for v in all_u[col]]

        # logarithmic scale
        if col in ('media_count', 'follower_count', 'following_count',
                   'following_tag_count', 'usertags_count', 'total_igtv_videos',
                   'interop_messaging_user_fbid', 'p', 'l', 'c'):
            scale = math.log(min([v for v in all_u[col] if v])) - 1
            scaled = [1 - math.log(v) / scale if v else 0.0 for v in all_u[col]]
            all_u[col] = scaled
            pass

        pass

    print_with_time(f'scale all columns to [0, 1]')


def feature_selection(all_u: pd.DataFrame):

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

    del all_u['pk']  # correlation with 'bot' == 0.92, look like cheat
    del all_u['interop_messaging_user_fbid']  # correlation with 'bot' == 0.93, look like cheat
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


def save_features_as_train_and_test(all_u: pd.DataFrame):

    X_train, _X_test = train_test_split(
        all_u,
        test_size=0.2,
        shuffle=True)

    train_store = pd.HDFStore(TRAIN_DATA_FILE)
    test_store = pd.HDFStore(TEST_DATA_FILE)

    train_store[DATAFRAME_NAME] = X_train
    test_store[DATAFRAME_NAME] = _X_test

    print_with_time('\nstore train and test dataframes in files')


all_users_dataframe: pd.DataFrame = read_accounts_from_json_to_dataframe()
feature_extraction(all_users_dataframe)
feature_selection(all_users_dataframe)

save_features_as_train_and_test(all_users_dataframe)