from config import EPOCH_COUNT, HIDDEN_NEURONS_COUNT,\
        LEARNING_RATE, LEARN_TYPE, LEARNING_ATTEMPTS_COUNT, \
        BATCH_SIZE, MOMENTUM, MODEL_SAVE_FILE, TRAIN_DATA_FILE, DATAFRAME_NAME

from net import InstaNet

# from collect import all_u, X_train, y_train  # _y_test, _X_test

import time
import random
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


def f16(t: torch.Tensor) -> np.float16:
    return np.float16(float(t))


def random_seed_initialization(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


train_store = pd.HDFStore(TRAIN_DATA_FILE, mode='r')
X_train = train_store[DATAFRAME_NAME]
train_store.close()
y_train = X_train['bot']
del X_train['bot']

X_train = X_train[sorted(X_train.columns)]  # make order the same for learning and inference
X_train = torch.FloatTensor(X_train.to_numpy())  # maybe HalfTensor
y_train = torch.LongTensor(y_train.to_numpy())  # maybe CharTensor or BoolTensor

X_train, X_validation, y_train, y_validation = train_test_split(
    X_train,
    y_train,
    test_size=0.3,
    shuffle=True)

# print(X_train.dtype, y_train.dtype, X_validation.dtype, y_validation.dtype)
# print(X_train.shape, y_train.shape, X_validation.shape, y_validation.shape)
#
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(torch.cuda.is_available())
# !nvidia-smi

train_accuracy_histories = []
val_accuracy_histories = []
train_loss_histories = []
val_loss_histories = []

net_weights = defaultdict(list)

for attempt in range(LEARNING_ATTEMPTS_COUNT):

    random_seed_initialization(int(time.time()))

    # loss = torch.nn.BCEWithLogitsLoss()  # TODO or CrossEntopyLoss() or BCELoss() ?
    loss = torch.nn.CrossEntropyLoss()

    insta_net = InstaNet(X_train.shape[1], HIDDEN_NEURONS_COUNT)

    if LEARN_TYPE == 'SGD':
        optimizer = torch.optim.SGD(insta_net.parameters(), lr=LEARNING_RATE,
                                    momentum=0)
    elif LEARN_TYPE == f'SGD_momentum_{MOMENTUM}':
        optimizer = torch.optim.SGD(insta_net.parameters(), lr=LEARNING_RATE,
                                    momentum=MOMENTUM)
    elif LEARN_TYPE == f'SGD_Nesterov_momentum_{MOMENTUM}':
        optimizer = torch.optim.SGD(insta_net.parameters(), lr=LEARNING_RATE,
                                    momentum=MOMENTUM, nesterov=True)
    elif LEARN_TYPE == f'ASGD':
        optimizer = torch.optim.ASGD(insta_net.parameters(), lr=LEARNING_RATE)
    elif LEARN_TYPE == 'Adam':
        optimizer = torch.optim.Adam(insta_net.parameters(), lr=LEARNING_RATE)

    insta_net = insta_net.to(device)
    X_validation = X_validation.to(device)
    y_validation = y_validation.to(device)

    train_accuracy_history = []
    val_accuracy_history = []
    train_loss_history = []
    val_loss_history = []

    start_time = time.time()
    for epoch in range(EPOCH_COUNT):
        order = np.random.permutation(len(X_train))

        epoch_train_loss = []
        epoch_train_accuracy = []

        for start_index in range(0, len(X_train), BATCH_SIZE):
            optimizer.zero_grad()

            batch_indexes = order[start_index:start_index + BATCH_SIZE]

            X_batch = X_train[batch_indexes].to(device)
            y_batch = y_train[batch_indexes].to(device)

            preds = insta_net.forward(X_batch)
            preds.squeeze_()  # TODO wtf ???

            loss_value = loss(preds, y_batch)
            loss_value.backward()

            epoch_train_loss.append(f16(loss_value))
            preds = preds.argmax(dim=1)
            epoch_train_accuracy.append(f16((preds == y_batch).float().mean()))

            optimizer.step()

        val_preds = insta_net.forward(X_validation)
        val_preds.squeeze_()

        val_loss_history.append(f16(loss(val_preds, y_validation)))
        train_loss_history.append(f16(sum(epoch_train_loss) / len(epoch_train_loss)))

        val_preds = val_preds.argmax(dim=1)
        val_accuracy = f16((val_preds == y_validation).float().mean())
        train_accuracy = f16(sum(epoch_train_accuracy) / len(epoch_train_accuracy))
        train_accuracy_history.append(train_accuracy)
        val_accuracy_history.append(val_accuracy)
        # print(f'epoch: {epoch} val accuracy: {val_accuracy}, train accuracy: {train_accuracy}')

    print(f'attempt: {attempt} '
          f'epochs: {EPOCH_COUNT} '
          f'val accuracy: {val_accuracy_history[-1]:0.3f}, '
          f'train accuracy: {train_accuracy_history[-1]:0.3f} '
          f'time: {np.float16(time.time() - start_time):0.2f} ')

    # test_preds = insta_net.forward(_X_test)
    # test_preds.squeeze_()
    # test_accuracy = f16((test_preds == _y_test).float().mean())
    #
    # print(f'test accuracy: {test_accuracy}')

    for name, param in insta_net.named_parameters():
        # print(name, param)
        net_weights[name].append(param)

    train_loss_histories.append(train_loss_history)
    val_loss_histories.append(val_loss_history)

    train_accuracy_histories.append(train_accuracy_history)
    val_accuracy_histories.append(val_accuracy_history)


    torch.save(insta_net.state_dict(), MODEL_SAVE_FILE)

pass


# X_train = train_store[DATAFRAME_NAME]
# train_store.close()
# del X_train['bot']

# for key, values_list in net_weights.items():
#     matrix_numpy = np.array([v.detach().numpy().squeeze() for v in values_list])[0]
#     if matrix_numpy.ndim == 2 and matrix_numpy.shape[1] == len(X_train.columns):
#         df = pd.DataFrame(matrix_numpy, columns=X_train.columns)
#         description = df.describe(include='all')
#         description = description.sort_values(axis=1, by='mean', key=lambda x: -abs(x)).transpose()
#         pass

#     # for value in values_list:
#     #     value_numpy = value.detach().numpy()
#     #     value_dataframe = pd.DataFrame(value_numpy)
#     #     pass
# pass
