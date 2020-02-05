import matplotlib.pyplot as plt
from configuration import *
from lstm import *


def plot_time_series_data(matrix, label, pred):
    fig, ax = plt.subplots(nrows=5, ncols=3, sharex=True, figsize=(12, 10), constrained_layout=True)
    true_label = "apoptosis" if label == 1.0 else "non-apoptosis"
    pred_label = "apoptosis" if pred == 1 else "non-apoptosis"

    for idx, (row, ff) in enumerate(zip(matrix.T, FEATURESNAME_WF)):
        time_point = [x * 6 for x in range(TIMELENGTH)]
        i = idx % 5
        j = idx // 5
        ax[i][j].plot(time_point, row)
        for x, y in zip(time_point, row):
            ax[i][j].scatter(x, y, s=5, c='b')
        ax[i][j].set_title(ff)
        if i == 4 or (i == 1 and j == 2):
            ax[i][j].set_xlabel("time (min)")

    ax[2][2].axis("off")
    ax[3][2].axis("off")
    ax[4][2].axis("off")
    fig.suptitle("true label: "+true_label+"\npredicted label: "+pred_label, x=0.85, y=0.35)
    fig.show()




