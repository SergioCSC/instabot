from pathlib import Path

SLOW_MODE = 0  # 0 is fast (~5 min), 1 is middle (~0.5 hour), 2 is slow (~6 hours)
LEARNING_ATTEMPTS_COUNT = 1  # number of tries of learning. for debug, only last attempt is used
EPOCH_COUNT = 1500  # number of epochs of learning. Usually not greater than 1500
BATCH_SIZE = 100  # batch size in learning. Default value 100 seems to be ok
HIDDEN_NEURONS_COUNT = 5  # num of neurons in hidden layers
LEARNING_RATE = 3.0e-4  # rate of learning. 3.0e-4 is ok for Adam method
LEARN_TYPE = 'Adam'  # other type of learning are 'ASGD', 'SGD' and 2 'SGD' with momentum (see learning.py)
MOMENTUM = 0.9  # used in SGD with momentum and SGD with Nesterov momentum only

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
THIRD_PARTY_LIBRARIES_DIR = THIS_PYTHON_SCRIPT_DIR / 'third_party_models'
HUGGINGFACE_CACHE_DIR_OS_ENVIRONMENT_VAR = 'TRANSFORMERS_CACHE'
HUGGINGFACE_DIR = THIRD_PARTY_LIBRARIES_DIR / HUGGINGFACE_CACHE_DIR_OS_ENVIRONMENT_VAR

DATA_DIR = THIS_PYTHON_SCRIPT_DIR / 'data'

PARSED_ACCOUNTS_FILE = DATA_DIR / 'accounts.txt'
PARSED_POSTS_FILE = DATA_DIR / 'posts.txt'
PREPARED_ACCOUNTS_FILENAME = 'accounts_with_posts.json'

MODEL_DIR = THIS_PYTHON_SCRIPT_DIR / 'model'

TRAIN_DATA_FILE = MODEL_DIR / 'train_dataframe.h5'
TEST_DATA_FILE = MODEL_DIR / 'test_dataframe.h5'
FEATURES_DATA_FILE = MODEL_DIR / 'features.pickle'
DEPENDED_FEATURES_DATA_FILE = MODEL_DIR / 'depended_features.pickle'
MODEL_SAVE_FILE = MODEL_DIR / 'model.pt'
