from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from configuration import *
from lstm.config import config_dict

def step_decay(epoch):
    prolong = 10
    if epoch < 10:
        return config_dict["init learning rate"]
    else:
        return config_dict["init learning rate"] * 0.5 ** ((epoch-prolong) // 4)


def getCallback(reduce_mode, lr_scheduler_mode):
    callbacks = []
    checkpoint = ModelCheckpoint(DEEP_LSTM_ROOT + '\\model_0202_lstm_test.h5',
                                 monitor="val_loss",
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min',
                                 save_weights_only=False,
                                 period=1)

    reduce = ReduceLROnPlateau(monitor="val_loss",
                               factor=0.5,
                               patience=3,
                               verbose=1,
                               mode='auto',
                               cooldown=1)

    lrs = LearningRateScheduler(step_decay)

    callbacks.append(checkpoint)

    if reduce_mode:
        callbacks.append(reduce)
    if lr_scheduler_mode:
        callbacks.append(lrs)

    return callbacks

