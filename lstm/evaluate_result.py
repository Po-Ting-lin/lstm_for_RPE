import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc


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


def evaluate_performance(model, X_test, y_test):
    predictions = model.predict(X_test)
    FPR, TPR, threshold = roc_curve(y_test, predictions)
    return auc(FPR, TPR)

