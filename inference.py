from nn_config import HIDDEN_NEURONS_COUNT, MODEL_SAVE_FILE, TEST_DATA_FILE, DATAFRAME_NAME
from net import InstaNet

import torch
import pandas as pd

test_store = pd.HDFStore(TEST_DATA_FILE)
X_test = test_store[DATAFRAME_NAME]
y_test = X_test['bot']
del X_test['bot']

X_test = torch.FloatTensor(X_test.to_numpy())  # maybe FloatTensor
# y_train = torch.LongTensor(y_train.to_numpy())  # maybe CharTensor or BoolTensor
y_test = torch.LongTensor(y_test.to_numpy())  # maybe CharTensor or BoolTensor

insta_net = InstaNet(X_test.shape[1], HIDDEN_NEURONS_COUNT)
insta_net.load_state_dict(torch.load(MODEL_SAVE_FILE))
insta_net.eval()

# preds = insta_net.forward(X_test)
preds = insta_net.inference(X_test)

pass