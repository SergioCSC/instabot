from pathlib import Path


def csv2json(csv_filepath: Path, bot_value: int) -> str:
    assert csv_filepath.suffix == '.csv'
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


if __name__ == '__main__':
    usernames = ('_alexandra_arch', '43', 'alinkamoon',
                 'cha_food', 'smagincartoonist')
    THIS_PYTHON_SCRIPT_DIR = Path(__file__).resolve().parent

    for username in usernames:
        USERS_DIR = THIS_PYTHON_SCRIPT_DIR / 'parsed_users' / 'bots_business_april' \
                    / '21_april' / 'non_bots' / username / 'users'

        csv2json(USERS_DIR / 'users_output.csv', 0)
