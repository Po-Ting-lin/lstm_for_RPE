import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from visualize.evaluation import show_result
from lstm import *


def show_final_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(15 ,5))
    ax[0].set_title('loss')
    ax[0].set_xlabel('epochs')
    ax[0].set_ylabel('loss')
    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax[0].legend()


    ax[1].set_title('acc')
    ax[1].set_xlabel('epochs')
    ax[1].set_ylabel('%')
    ax[1].plot(history.epoch, history.history["acc"], label="Train acc")
    ax[1].plot(history.epoch, history.history["val_acc"], label="Validation acc")
    ax[1].legend()


def evaluate_model_performance(predictions, X_test, y_test):
    FPR, TPR, threshold = roc_curve(y_test, predictions)
    return auc(FPR, TPR)


def probabilities2label(target, t):
    return np.array(list(map(int, target > t)))


def evaluation(predictions, y_test):
    def test_this_threshold(t):
        prediction_label = probabilities2label(predictions, t)
        return show_result(y_test, prediction_label, print_mode=False)
    max_acc, max_t = 0, 0
    for i in np.arange(0.05, 1, 0.05):
        A = test_this_threshold(i)[0]
        if A > max_acc:
            max_acc, max_t = A, i

    prediction_label = probabilities2label(predictions, max_t)
    show_result(y_test, prediction_label)
    return prediction_label



