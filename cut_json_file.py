from pathlib import Path

from nn_config import ACCOUNTS_JSONS_DIR

THIS_PYTHON_SCRIPT_DIR = Path(__file__).resolve().parent
USERS_DIR = ACCOUNTS_JSONS_DIR  # THIS_PYTHON_SCRIPT_DIR / 'parsed_users' / 'bots_business_april' / 'users'
filename = 'users_output_converted_marked.json'


def cut_json_file(lines_count: int):
    with open(USERS_DIR / filename, encoding='utf8') as in_:
        with open(USERS_DIR / f'{lines_count}_{filename}', 'w', encoding='utf8') as out:
            i = 0
            for line in in_:
                i += 1
                if i > lines_count:
                    # line = line.strip()[:-1] + '\n]'
                    line = line + '\n]'
                out.write(line)
                if i > lines_count:
                    break


cut_json_file(lines_count=1)