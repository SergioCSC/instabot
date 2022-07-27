import sys

from parser_cfg import *

import json
from collections import defaultdict
from collections import OrderedDict
from pathlib import Path


def parser_im_post_2_dict(parser_im_posts_filepath: Path) -> dict:
    def post_2_dict(post: str) -> dict:
        chunks = post.split(':')
        assert len(chunks) == 8
        d = OrderedDict()
        d['post_id'] = chunks[0]
        user_id = chunks[2]
        user_login = chunks[3]
        d['post_login'] = f'0000000000000000000_{user_id}'
        d['post_url'] = f'https://www.instagram.com/p/{d["post_id"]}/'
        d['photos_url'] = f'["http://127.0.0.1:8000/api/photo/{d["post_id"]}__{user_id}/"]'
        d['text'] = chunks[1] if chunks[1] != '0' else ''
        d['likes_count'] = int(chunks[4]) if chunks[4] else 0
        d['comments_count'] = int(chunks[5]) if chunks[5] else 0
        views_count = int(chunks[6]) if chunks[6] else 0
        date_and_time = chunks[7].split('_')
        d['time'] = date_and_time[1]
        d['date'] = date_and_time[0]
        d['user_login'] = user_login
        ...
        return d

    def split_line_into_posts(line: str):
        def get_posts(part: str):
            if part in ('', '\n', '\\n'):
                return []
            split_patterns = "\', \'", '''\", \'''', '''\', \"''', '''\", \"'''
            len_split_pattern = 4
            split_indexes = [part.find(p) for p in split_patterns if part.find(p) >= 23]
            if not split_indexes:
                return [part]
            split_indexes.sort()
            for split_index in split_indexes:
                pre_split = part[split_index-23:split_index]
                if pre_split.count('_') != 1 or pre_split.count('.') != 3 \
                        or pre_split.count(':') not in (1, 2, 3, 4):
                    continue
                if not pre_split.replace(':', '').replace('.', '').replace('_', '').isdigit():
                    continue
                return [part[:split_index]] + get_posts(part[split_index + len_split_pattern:])
            return [part]

        result = []
        parts = line.split('\\n')
        for part in parts:
            result.extend(get_posts(part))
        return result

    if not Path(parser_im_posts_filepath).is_file():
        return {}
    with open(parser_im_posts_filepath, encoding='utf-8') as f:
        result = defaultdict(dict)
        for line in f:
            if line in ('', '\n', '\\n'):
                continue
            line = line[2:-3]
            posts = split_line_into_posts(line)
            pass
            for i, post in enumerate(posts):
                post_dict = post_2_dict(post)
                user_login = post_dict['user_login']
                post_id = post_dict['post_id']
                del post_dict['user_login']
                result[user_login][post_id] = post_dict
        return result


def parser_im_accounts_2_json(parser_im_accounts_filepath: Path, posts: dict,
                              accounts_with_bot_values: dict[str, int]) -> str:
    def get_username(user_: str) -> str:
        user_ = user_[user_.find('username') + 10:]
        user_ = user_.lstrip()[1:]
        username_ = user_[:user_.find('"')]
        return username_

    def add_posts(user_: str) -> str:
        user_ = user_.replace("\\'", "'")  # TODO why?
        user_login = get_username(user_)
        user_posts = list(posts[user_login].values()) if user_login in posts else []
        post_str = json.dumps(user_posts, ensure_ascii=False)
        post_str = post_str.replace('"[\\"', '["')
        post_str = post_str.replace('/\\"]"', '"]')
        user_ = user_[:-1] + ', "posts": ' + post_str + '}'
        return user_

    parsed_users = set()
    output_json_filepath = PREPARED_ACCOUNTS_FILE
    with open(parser_im_accounts_filepath, encoding='utf8') as in_:
        with open(output_json_filepath, 'w', encoding='utf8') as out:
            out.write('[\n  ')
            for line_num, line in enumerate(in_):
                if line in ('', '\n', '\\n'):
                    continue
                user_end = '},"status":"ok"}\\n'
                users = line.split(user_end)
                user_end = user_end[:-2]
                for i in range(len(users) - 1):
                    users[i] += user_end
                for user_num, user_str in enumerate(users):
                    username = get_username(user_str)
                    if username in parsed_users:
                        continue
                    else:
                        parsed_users.add(username)
                    assert (not accounts_with_bot_values) or (username in accounts_with_bot_values)
                    bot = accounts_with_bot_values[username] if accounts_with_bot_values else -1
                    user_str = user_str[user_str.find(':') + 1:]
                    user_str = user_str[user_str.find(':') + 1:]
                    start = '{"user": {"pk": '
                    if user_str.startswith(start):
                        user_str = f'{{"bot": {bot}, "pk": ' + user_str[len(start):]
                    start = '{"user":{"pk":'
                    if user_str.startswith(start):
                        user_str = f'{{"bot":{bot},"pk":' + user_str[len(start):]
                    user_str = user_str.replace('}, \"status\": \"ok\"}',
                                         ', \"status\": \"ok\"}', 1)
                    user_str = user_str.replace('},\"status\":\"ok\"}',
                                         ',\"status\":\"ok\"}', 1)

                    end = "']\n"
                    if user_str.endswith(end):
                        user_str = user_str[:-len(end)]

                    user_str = add_posts(user_str)

                    if line_num != 0 or user_num != 0:
                        user_str = ', ' + user_str

                    user_str = user_str.replace('\\\\"', '\\"')
                    user_str = user_str.replace('\\xad', 'xxad')
                    user_str = user_str.replace('\\U0001', 'U0001')
                    out.write(user_str + '\n')

            out.write(']')
    return output_json_filepath


def get_accounts_with_bot_values(account_list_files: list[str]) -> dict[str, int]:
    result = {}
    for account_list_file in account_list_files:
        with open(account_list_file) as f:
            for line in f:
                line = line.strip()
                if line in ('', '\n', '\\n') or line.startswith('https://'):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    print(f'every line of {account_list_file} must contain bot_value and username, separated by space')
                    raise AssertionError
                assert len(parts) >= 2
                assert parts[0].isdigit()
                result[parts[1]] = int(parts[0])
    return result


def main():
    account_list_files = sys.argv[1:]
    accounts_with_bot_values = get_accounts_with_bot_values(account_list_files)
    posts: dict = parser_im_post_2_dict(Path(PARSED_POSTS_FILE))
    out_file_path = parser_im_accounts_2_json(Path(PARSED_ACCOUNTS_FILE), posts, accounts_with_bot_values)


if __name__ == '__main__':
    main()
