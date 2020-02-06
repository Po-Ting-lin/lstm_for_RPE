from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from configuration import *


def getCallback(reduce_mode):
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

    callbacks.append(checkpoint)
    if reduce_mode:
        callbacks.append(reduce)
    return callbacks

