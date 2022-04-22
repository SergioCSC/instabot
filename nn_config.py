LEARNING_ATTEMPTS_COUNT = 10
EPOCH_COUNT = 300
BATCH_SIZE = 100
HIDDEN_NEURONS_COUNT = 5
LEARNING_RATE = 3.0e-4
LEARN_TYPE = f'Adam'
MOMENTUM = 0.9

TRAIN_DATA_FILE = 'train_dataframe.h5'
TEST_DATA_FILE = 'test_dataframe.h5'
MODEL_SAVE_FILE = 'state_dict_model.pt'

DATAFRAME_NAME = 'features_and_target'

from pathlib import Path
THIS_PYTHON_SCRIPT_DIR = Path(__file__).resolve().parent
ACCOUNTS_JSONS_DIR = THIS_PYTHON_SCRIPT_DIR / 'parsed_users' / 'bots_business_april' / 'users'
ACCOUNTS_JSONS_DIR_2 = THIS_PYTHON_SCRIPT_DIR / 'parsed_users' / 'bots_business_april' \
                       / '21_april' / 'non_bots' / '_alexandra_arch' / 'users'
