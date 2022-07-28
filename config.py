from pathlib import Path

LEARNING_ATTEMPTS_COUNT = 1
EPOCH_COUNT = 1500
BATCH_SIZE = 100
HIDDEN_NEURONS_COUNT = 5
LEARNING_RATE = 3.0e-4
LEARN_TYPE = 'Adam'
MOMENTUM = 0.9

DATAFRAME_NAME = 'features_and_target'
FEATURES_NAME = 'features'

BOT_COL = 'bot'
DETECTION = 'detected_bot'
SAVED_PK = 'saved_pk'
SAVED_UN = 'saved_username'

LANG_UNKNOWN = 'unknown'
COMMON_LANGS = ('en', 'ru', 'zh', 'hi', 'es', 'ar', 'bn', 'fr', 'pt', 'ur',
                'id', 'de', 'ja', 'it', LANG_UNKNOWN)

ALL_SENTIMENTS_RU = ('neutral', 'positive', 'negative')  #, 'speech', 'skip')
ALL_SENTIMENTS_EN = ('neu', 'pos', 'neg')
ALL_SENTIMENTS_MUL = ('Neutral', 'Positive', 'Negative')

THIS_PYTHON_SCRIPT_DIR = Path(__file__).resolve().parent

LEARNING_DATASETS_DIR = THIS_PYTHON_SCRIPT_DIR / 'learning_datasets'
ACCOUNTS_JSONS_DIR = THIS_PYTHON_SCRIPT_DIR / 'parsed_users' / 'bots_business_april' / 'users'
THIRD_PARTY_LIBRARIES_DIR = THIS_PYTHON_SCRIPT_DIR / 'third_party_models'
HUGGINGFACE_CACHE_DIR_OS_ENVIRONMENT_VAR = 'TRANSFORMERS_CACHE'
HUGGINGFACE_DIR = THIRD_PARTY_LIBRARIES_DIR / HUGGINGFACE_CACHE_DIR_OS_ENVIRONMENT_VAR

DATA_DIR = THIS_PYTHON_SCRIPT_DIR / 'data'

PARSED_ACCOUNTS_FILE = DATA_DIR / 'accounts.txt'
PARSED_POSTS_FILE = DATA_DIR / 'posts.txt'
PREPARED_ACCOUNTS_FILE = DATA_DIR / 'prepared_accounts.json'

MODEL_DIR = THIS_PYTHON_SCRIPT_DIR / 'model'

TRAIN_DATA_FILE = MODEL_DIR / 'train_dataframe.h5'
TEST_DATA_FILE = MODEL_DIR / 'test_dataframe.h5'
FEATURES_DATA_FILE = MODEL_DIR / 'features.pickle'
DEPENDED_FEATURES_DATA_FILE = MODEL_DIR / 'depended_features.pickle'
MODEL_SAVE_FILE = MODEL_DIR / 'model.pt'
