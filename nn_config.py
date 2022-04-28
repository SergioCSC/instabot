LEARNING_ATTEMPTS_COUNT = 1
EPOCH_COUNT = 1500
BATCH_SIZE = 100
HIDDEN_NEURONS_COUNT = 5
LEARNING_RATE = 3.0e-4
LEARN_TYPE = f'Adam'
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

from pathlib import Path
THIS_PYTHON_SCRIPT_DIR = Path(__file__).resolve().parent

LEARNING_DATASETS_DIR = THIS_PYTHON_SCRIPT_DIR / 'learning_datasets'

ACCOUNTS_JSONS_DIR = THIS_PYTHON_SCRIPT_DIR / 'parsed_users' / 'bots_business_april' / 'users'
# ACCOUNTS_JSONS_DIR_2 = THIS_PYTHON_SCRIPT_DIR / 'parsed_users' / 'bots_business_april' \
#                        / '21_april' / 'non_bots' / '_alexandra_arch' / 'users'
# ACCOUNTS_JSONS_DIR_3 = THIS_PYTHON_SCRIPT_DIR / 'parsed_users' / 'bots_business_april' \
#                        / '21_april' / 'non_bots' / '43' / 'users'
# ACCOUNTS_JSONS_DIR_4 = THIS_PYTHON_SCRIPT_DIR / 'parsed_users' / 'bots_business_april' \
#                        / '21_april' / 'non_bots' / 'cha_food' / 'users'
# ACCOUNTS_JSONS_DIR_5 = THIS_PYTHON_SCRIPT_DIR / 'parsed_users' / 'bots_business_april' \
#                        / '21_april' / 'non_bots' / 'smagincartoonist' / 'users'
# ACCOUNTS_JSONS_DIR_6 = THIS_PYTHON_SCRIPT_DIR / 'parsed_users' / 'bots_business_april' \
#                        / '21_april' / 'non_bots' / 'alinkamoon' / 'users'
# ACCOUNTS_JSONS_DIR_7 = THIS_PYTHON_SCRIPT_DIR / 'parsed_users' / 'bots_business_april'
#
# BOTS_JSONS_DIR = THIS_PYTHON_SCRIPT_DIR / 'parsed_users' / 'bots_business_april' \
#                        / '21_april' / 'bots' / 'manually' / 'users'
#
# BOTS_JSONS_DIR_2 = THIS_PYTHON_SCRIPT_DIR / 'parsed_users' / 'bots_business_april' \
#                        / '21_april' / 'bots' / 'insta_accs' / 'users'