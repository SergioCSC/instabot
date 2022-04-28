from nn_config import HIDDEN_NEURONS_COUNT, MODEL_SAVE_FILE, TEST_DATA_FILE, DATAFRAME_NAME, SAVED_PK, SAVED_UN, \
    BOT_COL, DETECTION
import collect
from net import InstaNet

from sklearn.metrics import classification_report
import torch
import pandas as pd
import numpy as np

import sys
from pathlib import Path


inference_accounts_filepath = Path(sys.argv[1]) if len(sys.argv) > 1 else None
if inference_accounts_filepath:
    collect.collect_and_save_features(inference_accounts_filepath)
    print(f'inference accounts file path: {inference_accounts_filepath}')

test_store = pd.HDFStore(TEST_DATA_FILE, mode='r')
X_test = test_store[DATAFRAME_NAME]
test_store.close()

y_test_col = X_test['bot']
saved_pk_col = X_test[SAVED_PK]
saved_username_col = X_test[SAVED_UN]

del X_test['bot']
del X_test[SAVED_PK]
del X_test[SAVED_UN]

X_test = X_test[sorted(X_test.columns)]    # make order the same for learning and inference
X_test_tensor = torch.FloatTensor(X_test.to_numpy())  # maybe FloatTensor
# y_train = torch.LongTensor(y_train.to_numpy())  # maybe CharTensor or BoolTensor
y_test_tensor = torch.LongTensor(y_test_col.to_numpy())  # maybe CharTensor or BoolTensor

insta_net = InstaNet(X_test_tensor.shape[1], HIDDEN_NEURONS_COUNT)
insta_net.load_state_dict(torch.load(MODEL_SAVE_FILE))
insta_net.eval()

# preds = insta_net.forward(X_test_tensor)
preds = insta_net.inference(X_test_tensor)
preds = preds.argmax(dim=1)
accuracy = ((preds == y_test_tensor).float().mean())
if y_test_tensor[0] != -1:
    print(f'accuracy: {np.float16(float(accuracy))}')

if len(set(int(x) for x in y_test_tensor)) == 3:
    report = classification_report(y_test_tensor, preds, target_names=['human', 'bots', 'business'])
    print(report)

print(f'predictions: {preds.tolist()}')

additional_columns = pd.DataFrame({BOT_COL: y_test_col,
                                   DETECTION: preds,
                                   SAVED_PK: saved_pk_col,
                                   SAVED_UN: saved_username_col},
                                  index=X_test.index)

X_test_extended = pd.concat([additional_columns, X_test], axis=1)

print(X_test_extended.to_string(columns=(SAVED_PK, SAVED_UN, BOT_COL, DETECTION),
                                index=False))

# print(f'correlation of bot and detected bot: '
#       f'{X_test_with_detected_bot[["bot", "detected_bot"]].corr(method="pearson")}')

# result_json = {}

# for
