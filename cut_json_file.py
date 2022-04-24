from pathlib import Path

from nn_config import ACCOUNTS_JSONS_DIR_5

THIS_PYTHON_SCRIPT_DIR = Path(__file__).resolve().parent
USERS_DIR = ACCOUNTS_JSONS_DIR_5  # THIS_PYTHON_SCRIPT_DIR / 'parsed_users' / 'bots_business_april' / 'users'
filename = 'users_output_converted_marked.json'
lines_number = 16

with open(USERS_DIR / filename, encoding='utf8') as in_:
    with open(USERS_DIR / f'{lines_number}_{filename}', 'w', encoding='utf8') as out:
        i = 0
        for line in in_:
            i += 1
            if i > lines_number:
                # line = line.strip()[:-1] + '\n]'
                line = line + '\n]'
            out.write(line)
            if i > lines_number:
                break
