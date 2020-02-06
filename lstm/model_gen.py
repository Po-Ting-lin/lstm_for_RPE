from keras.models import Sequential
from keras.layers import Dense, Conv1D, LSTM, Dropout, Bidirectional
from keras.optimizers import Adam


def bulid_model(config):
    num_samples, num_time_steps, num_features = config["input shape"]
    model = Sequential()
    model.add(Conv1D(filters=num_features, kernel_size=3, strides=1, padding='valid'))
    model.add(Bidirectional(LSTM(config['units'], input_shape=(num_time_steps, num_features), dropout=config['dropout'])))
    model.add(Dropout(config['dropout']))
    model.add(Dense(1, activation='sigmoid'))

    a = Adam(lr=config["learning rate"], beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(loss='binary_crossentropy', optimizer=a, metrics=['accuracy'])
    return model

