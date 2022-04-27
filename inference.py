from sklearn.metrics import classification_report

import collect
from nn_config import HIDDEN_NEURONS_COUNT, MODEL_SAVE_FILE, TEST_DATA_FILE, DATAFRAME_NAME
from net import InstaNet

import sys
from pathlib import Path
import torch
import pandas as pd
import numpy as np


inference_accounts_filepath = Path(sys.argv[1]) if len(sys.argv) > 1 else None
if inference_accounts_filepath:
    collect.collect_and_save_features(inference_accounts_filepath)
    print(f'inference accounts file path: {inference_accounts_filepath}')

test_store = pd.HDFStore(TEST_DATA_FILE, mode='r')
X_test = test_store[DATAFRAME_NAME]
test_store.close()

y_test = X_test['bot']
del X_test['bot']

X_test = X_test[sorted(X_test.columns)]    # make order the same for learning and inference
X_test = torch.FloatTensor(X_test.to_numpy())  # maybe FloatTensor
# y_train = torch.LongTensor(y_train.to_numpy())  # maybe CharTensor or BoolTensor
y_test = torch.LongTensor(y_test.to_numpy())  # maybe CharTensor or BoolTensor

insta_net = InstaNet(X_test.shape[1], HIDDEN_NEURONS_COUNT)
insta_net.load_state_dict(torch.load(MODEL_SAVE_FILE))
insta_net.eval()

# preds = insta_net.forward(X_test)
preds = insta_net.inference(X_test)
preds = preds.argmax(dim=1)
accuracy = ((preds == y_test).float().mean())
print(np.float16(float(accuracy)))

if len(set(int(x) for x in y_test)) == 3:
    report = classification_report(y_test, preds, target_names=['human', 'bots', 'business'])
    print(report)

pass