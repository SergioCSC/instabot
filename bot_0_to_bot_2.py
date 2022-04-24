import json
from pathlib import Path


def bot0_to_bot2(json_filepath: Path, users: list) -> str:
    output_json_filepath = str(json_filepath)[:-len(json_filepath.suffix)] + '_bot_is_2.json'
    with open(json_filepath, encoding='utf8') as in_:
        with open(output_json_filepath, 'w', encoding='utf8') as out:
            for line in in_:
                for user in users:
                    user_str = f', "username": "{user}", "full_name": "'
                    if user_str in line[:100]:
                        line = line.replace(', {"bot": 0, "pk": ',
                                            ', {"bot": 2, "pk": ', 1)
                        break
                out.write(line)
    return output_json_filepath


def find_bot2(json_filepath: Path) -> list[str]:
    f = open(json_filepath, "r", encoding='utf-8')
    users = json.loads(f.read())
    for user in users:
        if user['bot'] == 2:
            yield user['username']


if __name__ == '__main__':
    THIS_PYTHON_SCRIPT_DIR = Path(__file__).resolve().parent
    JSON_WITH_BOTS_2_FILE_PATH = THIS_PYTHON_SCRIPT_DIR / 'parsed_users' / 'json_with_posts' \
                / 'alinkamoon_no_bots.json'

    users_with_bot_2 = list(find_bot2(JSON_WITH_BOTS_2_FILE_PATH))

    JSON_WITH_BOTS_0_ONLY_FILEPATH = THIS_PYTHON_SCRIPT_DIR / 'parsed_users' \
                                     / 'bots_business_april' / '21_april' / 'non_bots' \
                                     / 'alinkamoon' / 'users' / 'users_output_converted.json'

    # another_users_with_bot_2 = list(find_bot2(JSON_WITH_BOTS_0_ONLY_FILEPATH))
    # print(set(users_with_bot_2) - set(another_users_with_bot_2))

    bot0_to_bot2(JSON_WITH_BOTS_0_ONLY_FILEPATH, users_with_bot_2)
