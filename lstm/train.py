from keras.models import load_model
from sklearn.model_selection import KFold
from lstm import *
from configuration import *

X, y = data_generator()

auc_buf = []
kf = KFold(n_splits=3, random_state=None, shuffle=True)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = bulid_model()

    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=50,
                        batch_size=16,
                        callbacks=getCallback(),
                        verbose=0)

    model = load_model(DEEP_LSTM_ROOT + '\\model_0202_lstm_test.h5')

    area_under_curve = evaluate_performance(model, X_test, y_test)
    auc_buf.append(area_under_curve)

k_auc = np.mean(auc_buf)
print("AUC: {:.3f} ".format(k_auc))

