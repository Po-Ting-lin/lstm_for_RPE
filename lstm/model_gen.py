from keras.models import Sequential
from keras.layers import Dense, Conv1D, LSTM, Dropout, Bidirectional


def bulid_model(config):
    num_samples, num_time_steps, num_features = config["input shape"]
    model = Sequential()
    model.add(Conv1D(filters=num_features, kernel_size=3, strides=1, padding='valid'))
    model.add(Bidirectional(LSTM(config['units'], input_shape=(num_time_steps, num_features), dropout=config['dropout'])))
    model.add(Dropout(config['dropout']))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

