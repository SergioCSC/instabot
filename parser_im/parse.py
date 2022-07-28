from parser_cfg import *

import time
import sys
import requests
from datetime import datetime
from typing import NamedTuple, Union, Optional, Generator
from codetiming import Timer


SESSION = requests.Session()


class Job(NamedTuple):
    name: str
    func: callable
    data: any


class Output(NamedTuple):
    time: datetime
    type_: str
    task: str
    tid: int
    iteration: int
    status: str
    names: list[str]
    response: str
    url: str

    @classmethod
    def __str__(cls) -> str:
        return f'{cls.status:15} {cls.type_} {cls.wait = } {cls.tid = } {cls.task = }'

# class GetRequestParams(NamedTuple):
#     url: str
#     key: str
#     mode: str
#     type: str
#     links: str
#     web: str
#     name: str


def tprint(s: str):
    print(f'{datetime.now(): %H:%M}  {s}')


def send_request(params: dict) -> requests.Response:
    while True:
        try:
            response = SESSION.get(url=URL, params=params)
            response.encoding = "utf-8"
            return response
        except requests.exceptions.ConnectionError as e:
            tprint(f'send_request() exception: {e}. Try again')


def read_users_list(filename: str):
    with open(file=filename) as f:
        for line in f:
            line = line.strip()
            if line not in ('', '\n', '\\n') and not line.startswith('https://'):
                yield line.split()[1 if line.split()[0].isdigit() else 0]


def get_parser_im_task_status(tid: int) -> Optional[str]:
    params = {
        'key': KEY,
        'mode': 'status',
        'tid': tid,
    }
    response = send_request(params)
    if not response.text:
        return 'empty'
    try:
        d = response.json()
    except requests.exceptions.JSONDecodeError as e:
        return 'error_json'
    task_status = d.get('tid_status', 'no_status')
    return task_status


def get_parser_im_task_result(tid: int) -> str:
    params = {
        'key': KEY,
        'mode': 'result',
        'tid': tid,
    }
    response = send_request(params)
    return response.text


def get_parser_im_task_results(tids: list[int], params: dict, response: requests.Response) -> list[str]:
    type_ = 'A' if params['type'] == 'f1' else 'P'
    wait = 0
    while tids:
        wait += 1
        for tid in tids:
            task_status = get_parser_im_task_status(tid)
            output = f'{type_} {wait = } {tid = } GET_TASK_RESULTS'
            if task_status in ('completed', 'suspended'):  # completed or suspended
                task_result = get_parser_im_task_result(tid)
                if task_result:
                    tids.remove(tid)
                    yield task_result
                    continue
                task_status = f'{task_status}_0'
                tprint(f'{task_status:15} {output} {task_result = } "users:" {params["links"][:STR_CUT]} {response.url = }')
                return
            tprint(f'{task_status:15} {output} "users:" {params["links"][:STR_CUT]} {response.url = }')

            yield


def posts_info_get_parsed_users(accounts_infos: list[str], users: list[str]):
    return users  # TODO it's a dummy. Has to be implemented


def account_info_get_parsed_users(accounts_infos: list[str], users: list[str]):  # TODO get rid of 2nd argument
    for accounts_info_ in accounts_infos:
        while True:
            left_colon_i = accounts_info_.index(':')
            right_colon_i = accounts_info_.index(':', left_colon_i + 1)
            username = accounts_info_[left_colon_i + 1: right_colon_i]
            yield username
            end_pattern = '},"status":"ok"}'
            next_part_i = accounts_info_.find(f'{end_pattern}\n')
            if next_part_i == -1:
                assert accounts_info_.endswith(end_pattern) and accounts_info_.count(end_pattern) == 1
                break
            accounts_info_ = accounts_info_[next_part_i + 17:]


def account_info_make_http_get_params(users: list[str]):
    mode = 'create'
    type_ = 'f1'
    web = '0'
    act = '1'
    spec = '1,2'
    dop = '12'
    links = ','.join(users)
    name = f'se_test_mode_{mode}_type_{type_}_web_{web}_act_{act}_spec_{spec}_dop_{dop}'
    params = {
        'key': KEY,
        'mode': mode,
        'type': type_,
        'web': web,
        'act': act,
        'spec': spec,
        'dop': dop,
        'name': name,
        'links': links,
    }
    return params


def posts_info_make_http_get_params(users: list[str]):
    mode = 'create'
    type_ = 'p1'
    web = '0'
    act = '6'
    spec = '4,8,9'
    links = ','.join(users)
    name = f'se_test_mode_{mode}_type_{type_}_web_{web}_act_{act}_spec_{spec}'
    params = {
        'key': KEY,
        'mode': mode,
        'type': type_,
        'act': act,
        'spec': spec,
        'web': web,
        'name': name,
        'links': links,
    }
    return params


def get_accounts_or_posts_info(users: list[str], posts: bool = False):

    i = 0
    while True:
        i += 1
        get_params_f = posts_info_make_http_get_params if posts else account_info_make_http_get_params
        params = get_params_f(users)
        response = send_request(params)
        yield

        try:
            d = response.json()
        except requests.exceptions.JSONDecodeError as e:
            tprint(f'{e = }')
            yield
            continue
        if 'tid' in d and d['tid'].replace(',', '').isdigit():
            tids = [int(tid) for tid in d['tid'].split(',')]
            result = []
            gen_result = get_parser_im_task_results(tids, params, response)
            while True:
                try:
                    res = next(gen_result)
                    if res:
                        result.append(res)
                    else:
                        yield
                except StopIteration:
                    break

            get_parsed_users_f = posts_info_get_parsed_users if posts else account_info_get_parsed_users
            parsed_users = set(get_parsed_users_f(result, users))
            non_parsed_users = list(set(users) - parsed_users)
            if result and not non_parsed_users:
                tprint(f'SUCCESS         {"P" if posts else "A"} {i    = } '
                       f'tid = {tids[0] if len(tids) == 1 else tids} '
                       f'{response.text = } "users:" {params["links"][:STR_CUT]} {response.url = }')
                yield result
                return

            fail_text = f'{"P" if posts else "A"} {i    = } ' \
                        f'tid = {tids[0] if len(tids) == 1 else tids} '
            fail_text += f'parsed only {len(parsed_users)} out of {len(users)} '
            if 0 < len(parsed_users):
                fail_text = f'PARTIAL SUCCESS {fail_text} {non_parsed_users = } '
            else:
                fail_text = f'fail            {fail_text} {result = } '
            fail_text += f'{response.text = } "users:" {params["links"][:STR_CUT]} {response.url = }'
            tprint(fail_text)
            users = non_parsed_users
            yield result
            continue
        else:
            status = f'{d["tid"]:15}' if "tid" in d else f'{response.status_code:15}'
            error_text = f'{status} {"P" if posts else "A"} '
            error_text += f'{i    = } {response.text = } "users:" {params["links"][:STR_CUT]} {response.url = }'
            tprint(error_text)
            yield
            continue


def main():
    start = f'{datetime.now():%H_%M__%d_%m}'
    gens = []
    account_list_files = sys.argv[1:]
    # for f in ('instagram_bots_infl.txt',
    #           'instagtam_bots_cleared.txt',
    #           'bots_com_like_inst_cleared.txt'):
    for f in account_list_files:
        users = list(read_users_list(f))
        gens.append((PARSED_ACCOUNTS_FILE, get_accounts_or_posts_info(users=users, posts=False)))
        for i in range(0, len(users), POST_INFO_BATCH_SIZE):
            batch = users[i: i + POST_INFO_BATCH_SIZE]
            gens.append((PARSED_POSTS_FILE, get_accounts_or_posts_info(users=batch, posts=True)))

    with Timer(text='\nTotal time: {:.1f}'):
        while gens:
            for f, gen in gens:
                try:
                    info = next(gen)
                    if info:
                        with open(f'{f}', 'a', encoding='utf-8') as out:
                            out.write(f'{info}\n\n\n\n\n')
                except StopIteration as e:
                    gens.remove((f, gen))
                finally:
                    time.sleep(WAITING_TIME)

    SESSION.close()

    # for f, accounts_infos in accounts_info_dict.items():
    #     for accounts_info in accounts_infos:
    #         time_ = datetime.now().strftime("%H_%M___%d_%m_%Y")
    #         with open(f'{f}_{time_}.out', 'a', encoding='utf-8') as out:
    #             out.write(f'{accounts_info}\n\n\n\n\n')
    #
    # pass


if __name__ == '__main__':
    main()
