from learning import val_accuracy_histories
from learning import train_accuracy_histories
from learning import val_loss_histories
from learning import train_loss_histories
from learning import LEARN_TYPE
from learning import LEARNING_RATE
from learning import NEURONS_COUNT
from learning import BATCH_SIZE
from learning import LEARNING_ATTEMPTS_COUNT

import matplotlib
from matplotlib import pyplot as plt

# import matplotlib
matplotlib.rcParams['figure.figsize'] = (8, 8)
styles = (  # 'Solarize_Light2',
          # '_classic_test_patch',
          # '_mpl-gallery',
          'bmh',
          # 'classic',
          # 'dark_background',
          # 'seaborn-bright',
          # 'tableau-colorblind10'
          )
# plt.gca().set_ylim(0, 1)
for style in styles:
    matplotlib.style.use(style=style)
    for i in range(len(val_accuracy_histories)):
        plt.plot(val_accuracy_histories[i], label=f'attempt {i} validation accuracy')
    for i in range(len(val_accuracy_histories)):
        plt.plot(train_accuracy_histories[i], label=f'attempt {i} train accuracy')
    # plt.plot(train_accuracy_history, label=f'{LEARN_TYPE} lr={LEARNING_RATE} train accuracy')
    # plt.gca().set_ylim(0.0, 0.2)
    # plt.plot(val_loss_history, label=f'{LEARN_TYPE} lr={LEARNING_RATE} val loss');
    # plt.plot(train_loss_history, label=f'{LEARN_TYPE} lr={LEARNING_RATE} train loss');
    title = f'attempts = {LEARNING_ATTEMPTS_COUNT} ' \
            f'neurons = {NEURONS_COUNT} ' \
            f'batch size = {BATCH_SIZE} ' \
            f'method = {LEARN_TYPE} ' \
            f'learning rate = {LEARNING_RATE} '
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show()
    pass

pass
