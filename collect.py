from nn_config import TRAIN_DATA_FILE, TEST_DATA_FILE, DATAFRAME_NAME
from userpoststext import split_words
import userpostsinfo as upi

import statistics

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
model = fasttext.load_model('lid.176.ftz')


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


start_time = time.time()

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

print(f'read jsons: {time.time() - start_time}')

col = 'biography_with_entities'
all_u[col] = [len(v['entities']) for v in all_u[col]]

POSTS_COLUMN = 'posts'
users_posts = all_u[POSTS_COLUMN]  # [:10]
users_posts_lens = [[len(p['text']) for p in user_posts] for user_posts in users_posts]
users_total_lens = [sum(posts_lens) for posts_lens in users_posts_lens]
users_average_post_lens = [sum(pl) / len(pl) if pl else 0 for pl in users_posts_lens]
users_stdev_post_lens = [statistics.stdev(pl) if len(pl) > 1 else 0 for pl in users_posts_lens]

users_posts_emojis = [[[c for c in p['text'] if c in emoji.UNICODE_EMOJI['en']] for p in user_posts] for user_posts in users_posts]
users_emoji_percents = [[len([c for c in p['text'] if c in emoji.UNICODE_EMOJI['en']])/len(p['text']) for p in user_posts] for user_posts in users_posts]
users_emoji_average_percent = [sum(user_emoji_percents)/len(user_emoji_percents) if user_emoji_percents else 0 for user_emoji_percents in users_emoji_percents]

users_posts_langs = [Counter(model.predict(p['text'])[0][0][9:] for p in user_posts) for user_posts in users_posts]

users_vocabularies = [Counter(word for p in user_posts for word in split_words(p['text'].lower())) for user_posts in users_posts]
for user_vocabulary in users_vocabularies:
    del user_vocabulary['']
#
all_u['total_posts_length'] = users_total_lens
all_u['average_post_length'] = users_average_post_lens
all_u['stdev_posts_length'] = users_stdev_post_lens

all_u[POSTS_COLUMN] = [v if not (isinstance(v, float) and math.isnan(v)) else [] for v in all_u[POSTS_COLUMN]]

all_u[upi.POSTS_N] = [len(v) for v in all_u[POSTS_COLUMN]]
all_u[upi.LIKES_N] = [sum([int(p['likes_count']) for p in v]) for v in all_u[POSTS_COLUMN]]
all_u[upi.COMMENTS_N] = [sum([int(p['comments_count']) for p in v]) for v in all_u[POSTS_COLUMN]]


print(f'calc posts likes comments: {time.time() - start_time}')

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

print(f'fill list of UserPostInfo (=set of counters): {time.time() - start_time}')  # 6 sec with @dataclass of Counter's filling

# calibrate: v --> v / overall_v for all post, likes, comments values
for user_posts_info in users_posts_info:
    overall_value = getattr(user_posts_info, 'overall')
    for attr in user_posts_info.__annotations__:
        if attr != 'overall':
            attr_value = getattr(user_posts_info, attr)
            for k in overall_value:
                attr_value[k] = attr_value[k] / overall_value[k] if attr_value[k] else 0.0


print(f'calibrate: v --> v / overall_v for all post, likes, comments values: {time.time() - start_time}')

# calc variances
upi.fill_variances(users_posts_info)

print(f'calc variances: {time.time() - start_time}')

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


print(f'ndarray from counters: {time.time() - start_time}')  # 4 minutes with Dataframes filling

# make all_u columns from posts likes comments rates
# all_u[post_columns] = pd.DataFrame(users_post_ndarray_)  # bug: rows with same label
# all_u = pd.concat([all_u, pd.DataFrame(users_post_ndarray_)], axis=1)
# all_u.join(pd.DataFrame(users_post_ndarray_))
# all_u[post_columns[0]] = users_post_ndarray_[:, 0]
all_u[post_columns] = pd.DataFrame(users_post_ndarray_, index=all_u.index)  # bug: rows with same label
all_u = all_u.copy()  # defragmentation of dataframe

print(f'add ndarray to all_u: {time.time() - start_time}')

description = all_u.describe(include='all')  # .loc['unique', :]

print(f'all_u.descibe(): {time.time() - start_time}')

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


print(f'cells: list, dict, str --> int: {time.time() - start_time}')

# filter out columns with very small fraction of non-trivial values
for col in all_u:
    most_popular_popularity, _ = most_popular_list_value(list(all_u[col]))
    non_most_popular_values_fraction = 1 - most_popular_popularity / len(all_u)
    if non_most_popular_values_fraction < NON_TRIVIAL_VALUES_FRACTION_THRESHOLD:
        del all_u[col]

print(f'filter out columns with very small fraction of non-trivial values: {time.time() - start_time}')

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

print(f'scale all columns to [0, 1]: {time.time() - start_time}')

target = all_u['bot']

# corr = all_u.corr(method='pearson')
# shuffled_all_u = all_u.sample(frac=1)
# auto_correlations = sorted([(c, (shuffled_all_u[c]).autocorr()) for c in all_u], key=lambda p: p[1])

del all_u['external_lynx_url']  # correlation with 'external_url' == 1
del all_u['can_hide_public_contacts']  # correlation with 'can_hide_category' == 1
del all_u['charity_profile_fundraiser_info']  # correlation with 'is_business' == 1
del all_u['contact_phone_number']  # correlation with 'public_phone_number' == 1
del all_u['city_name']  # correlation with 'city_id' == 0.9998

# del all_u['is_eligible_for_smb_support_flow']  # correlation with 'is_interest_account' == 0.97
del all_u['posts']  # same as 'p_overall'
del all_u['city_id']  # correlation with latitude > 0.97
del all_u['public_phone_country_code']  # corr with public_phone_number > 0.97
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

del all_u['bot']

# low importance due to 1 neuron learning
# del all_u['full_name']
# del all_u['follower_count']
# del all_u['usertags_count']
# del all_u['hd_profile_pic_versions']
# del all_u['longitude']
# del all_u['public_phone_country_code']

print(f'del highly correlated columns: {time.time() - start_time}')
# corr = all_u.corr(method='pearson')

# print(min(corr))

X_train, _X_test, y_train, _y_test = train_test_split(
    all_u,
    target,
    test_size=0.1,
    shuffle=True)

X_train = torch.FloatTensor(X_train.to_numpy())  # maybe FloatTensor
# y_train = torch.LongTensor(y_train.to_numpy())  # maybe CharTensor or BoolTensor
y_train = torch.LongTensor(y_train.to_numpy())  # maybe CharTensor or BoolTensor

_X_test = torch.FloatTensor(_X_test.to_numpy())
# _y_test = torch.LongTensor(_y_test.to_numpy())
_y_test = torch.HalfTensor(_y_test.to_numpy())
