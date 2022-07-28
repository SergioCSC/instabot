from pathlib import Path

URL = f'https://parser.im/api.php'
# KEY = '***REMOVED***'  # hypepotok1
KEY = '***REMOVED***'  # potokpotok
POST_INFO_BATCH_SIZE = 4
WAITING_TIME = 1
STR_CUT = 100

PARSER_DIR = Path(__file__).resolve().parent
PARSED_ACCOUNTS_FILE = PARSER_DIR / 'accounts.txt'
PARSED_POSTS_FILE = PARSER_DIR / 'posts.txt'
PREPARED_ACCOUNTS_FILE = PARSER_DIR / 'prepared_accounts.json'
