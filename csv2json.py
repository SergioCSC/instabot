from pathlib import Path


def csv2json(csv_filepath: Path, bot_value: int) -> str:
    # assert csv_filepath.suffix == '.csv'
    output_json_filepath = str(csv_filepath)[:-len(csv_filepath.suffix)] + '_converted.json'
    with open(csv_filepath, encoding='utf8') as in_:
        with open(output_json_filepath, 'w', encoding='utf8') as out:
            out.write('[\n  ')
            line_number = 0
            for line in in_:
                start = '{"user": {"pk": '
                if line.startswith(start):
                    line = f'{{"bot": {bot_value}, "pk": ' + line[len(start):]
                start = '{"user":{"pk":'
                if line.startswith(start):
                    line = f'{{"bot":{bot_value},"pk":' + line[len(start):]
                line = line.replace('}, \"status\": \"ok\", \"posts\": [',
                                     ', \"status\": \"ok\", \"posts\": [', 1)
                line = line.replace('},\"status\":\"ok\",\"posts\":[',
                                     ',\"status\":\"ok\",\"posts\":[', 1)
                ends = ('},"status":"ok"}', '},"status":"ok"}\n',
                        '}, "status": "ok"}', '}, "status": "ok"}\n')
                for end in ends:
                    if line.endswith(end):
                        line = line[:-len(end)] + line[-len(end) + 1:]
                if line_number != 0:
                    line = ', ' + line
                out.write(line)
                line_number += 1

            out.write(']')
    return output_json_filepath


def parser_im_2_json(parser_im_filepath: Path, bot_value: int) -> str:
    # assert parser_im_filepath.suffix == '.csv'
    output_json_filepath = str(parser_im_filepath)[:-len(parser_im_filepath.suffix)] + '_converted.json'
    with open(parser_im_filepath, encoding='utf8') as in_:
        with open(output_json_filepath, 'w', encoding='utf8') as out:
            out.write('[\n  ')
            for line_num, line in enumerate(in_):
                if line == '\n':
                    continue
                user_end = '},"status":"ok"}\\n'
                users = line.split(user_end)
                user_end = user_end[:-2]
                for i in range(len(users) - 1):
                    users[i] += user_end
                for user_num, user in enumerate(users):
                    user = user[user.find(':') + 1:]
                    user = user[user.find(':') + 1:]
                    start = '{"user": {"pk": '
                    if user.startswith(start):
                        user = f'{{"bot": {bot_value}, "pk": ' + user[len(start):]
                    start = '{"user":{"pk":'
                    if user.startswith(start):
                        user = f'{{"bot":{bot_value},"pk":' + user[len(start):]
                    user = user.replace('}, \"status\": \"ok\"}',
                                         ', \"status\": \"ok\"}', 1)
                    user = user.replace('},\"status\":\"ok\"}',
                                         ',\"status\":\"ok\"}', 1)

                    end = "']\n"
                    if user.endswith(end):
                        user = user[:-len(end)]
                    if line_num != 0 or user_num != 0:
                        user = ', ' + user
                    out.write(user + '\n')

            out.write(']')
    return output_json_filepath


if __name__ == '__main__':
    # usernames = ('_alexandra_arch', '43', 'alinkamoon',
    #              'cha_food', 'smagincartoonist')
    # usernames = ('smagincartoonist', )
    THIS_PYTHON_SCRIPT_DIR = Path(__file__).resolve().parent
    #
    # for bot_folders in ('insta_accs', 'manually'):
    #     USERS_DIR = THIS_PYTHON_SCRIPT_DIR / 'parsed_users' / 'bots_business_april' \
    #                 / '21_april' / 'bots' / bot_folders / 'users'
    #     csv2json(USERS_DIR / 'users_output.csv', 1)
    #
    # for username in usernames:
    #     USERS_DIR = THIS_PYTHON_SCRIPT_DIR / 'parsed_users' / 'bots_business_april' \
    #                 / '21_april' / 'non_bots' / username / 'users'
    #
    #     csv2json(USERS_DIR / 'users_output.csv', 0)

    USERS_DIR = THIS_PYTHON_SCRIPT_DIR / 'learning_datasets'
    # f = THIS_PYTHON_SCRIPT_DIR / 'parser_im' / 'bots_com_like_inst_cleared.txt_01_32__15_07_a_.out'
    f = THIS_PYTHON_SCRIPT_DIR / 'parser_im' / 'instagram_bots_infl.txt_18_11__15_07_a_.out'
    parser_im_2_json(f, -666)

