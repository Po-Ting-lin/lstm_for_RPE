from keras.models import load_model
from sklearn.model_selection import KFold
from lstm import *
from configuration import *
import random
import time
import matplotlib.pyplot as plt


# data generator
X, y = data_generator()

# training
auc_buf = []
kf = KFold(n_splits=3, random_state=None, shuffle=True)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    input_shape = X.shape
    model = bulid_model(input_shape)

    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=50,
                        batch_size=8,
                        callbacks=getCallback(),
                        verbose=0)

    model = load_model(DEEP_LSTM_ROOT + '\\model_0202_lstm_test.h5')
    predictions = model.predict(X_test)

    area_under_curve = evaluate_model_performance(predictions, X_test, y_test)
    preds = evaluation(predictions, y_test)
    auc_buf.append(area_under_curve)

    # for g in range(5):
    #     random.seed(int(time.time() * 100) % 100)
    #     idx = random.randint(0, len(y_test)-1)
    #     plot_time_series_data(X_test[idx], y_test[idx], preds[idx])


k_auc = np.mean(auc_buf)
print("AUC: {:.3f} ".format(k_auc))