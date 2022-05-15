from pathlib import Path

LEARNING_ATTEMPTS_COUNT = 1
EPOCH_COUNT = 1500
BATCH_SIZE = 100
HIDDEN_NEURONS_COUNT = 5
LEARNING_RATE = 3.0e-4
LEARN_TYPE = 'Adam'
MOMENTUM = 0.9

TRAIN_DATA_FILE = 'train_dataframe.h5'
TEST_DATA_FILE = 'test_dataframe.h5'
FEATURES_DATA_FILE = 'features.pickle'
DEPENDED_FEATURES_DATA_FILE = 'depended_features.pickle'
MODEL_SAVE_FILE = 'model.pt'

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
ALL_SENTIMENTS_EN = ('neu', 'pos', 'neg')  #, 'speech', 'skip')

THIS_PYTHON_SCRIPT_DIR = Path(__file__).resolve().parent

LEARNING_DATASETS_DIR = THIS_PYTHON_SCRIPT_DIR / 'learning_datasets'
ACCOUNTS_JSONS_DIR = THIS_PYTHON_SCRIPT_DIR / 'parsed_users' / 'bots_business_april' / 'users'
THIRD_PARTY_LIBRARIES_DIR = THIS_PYTHON_SCRIPT_DIR / 'third_party_models'
