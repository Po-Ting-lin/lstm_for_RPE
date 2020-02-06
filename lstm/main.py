from keras.models import load_model
from sklearn.model_selection import KFold
from lstm import *
from configuration import *
import random
import time
import matplotlib.pyplot as plt
from keras.utils import plot_model


# data generator
X, y = data_generator()


# config
config_dict = {"input shape": X.shape,
               "units": 20,
               "dropout": 0.5,
               "batch size": 10,
               "epochs": 50,
               "learning rate": 0.001,
               "reduce learning rate": 0}

auc_lr = []
for lr in [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1]:

    print(lr)
    config_dict["learning rate"] = lr

    # training
    auc_buf = []
    kf = KFold(n_splits=3, random_state=None, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = bulid_model(config_dict)


        plot_model(model, to_file='model.png', show_shapes=True)

        history = model.fit(X_train, y_train,
                            validation_data=(X_test, y_test),
                            epochs=config_dict["epochs"],
                            batch_size=config_dict['batch size'],
                            callbacks=getCallback(config_dict["reduce learning rate"]),
                            verbose=0)

        model.summary()

        model = load_model(DEEP_LSTM_ROOT + '\\model_0202_lstm_test.h5')
        predictions = model.predict(X_test)

        area_under_curve = evaluate_model_performance(predictions, y_test, plot_mode=True)
        preds = evaluation(predictions, y_test)
        auc_buf.append(area_under_curve)

        # for g in range(5):
        #     random.seed(int(time.time() * 100) % 100)
        #     idx = random.randint(0, len(y_test)-1)
        #     plot_time_series_data(X_test[idx], y_test[idx], preds[idx])

        show_final_history(history)


    k_auc = np.mean(auc_buf)
    print("AUC: {:.3f} ".format(k_auc))
    auc_lr.append(k_auc)


plt.figure()
plt.plot([0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1],[0.914, 0.968, 0.972, 0.984, 0.966, 0.986, 0.981, 0.94, 0.813])
plt.xticks([0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1])
plt.show()
