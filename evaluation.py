import numpy as np
import matplotlib.pyplot as plt

y_true = [1,2,3,4,1,2,3,4]
y_pred = [1,1,3,4,1,2,2,4]


def confusion_matrix(y_true, y_pred,plot=False,title=None):
    confusion_matrix = np.zeros((4, 4))
    if type(y_true[0])!= int:
        y_true = [int(x) for x in y_true.tolist()]
    if type(y_pred[0])!= int:
        y_pred = [int(x) for x in y_pred.tolist()]
    for x, y in zip(y_pred, y_true):
        confusion_matrix[x - 1][y - 1] += 1


    if plot==True:
        fig, ax = plt.subplots(1,1)
        plt.imshow(confusion_matrix)
        plt.xlabel("predicted")
        plt.ylabel("true")
        if title:
            plt.title(title)
        label_list = ['1','2', '3', '4']

        ax.set_xticks([0,1,2,3])
        ax.set_yticks([0, 1, 2, 3])

        ax.set_xticklabels(label_list)
        ax.set_yticklabels(label_list)

        for i in range(4):
            for j in range(4):
                text = ax.text(i,j, int(confusion_matrix[i][j]),
                               ha="center", va="center", color="w")
        plt.show()
    return confusion_matrix

"""def avg_recall_precision(confusion_matrix):
    N = len(confusion_matrix)

    recall = {}
    precision = {}
    for
    for room in range(N):
        true_pos = confusion_matrix[room][room]
        recall[room] = true_pos / sum(confusion_matrix[:][room])
        precision[room] = true_pos / sum(confusion_matrix[room][:])
    return sum(recall.values())/N, sum(precision.values()) / N
"""
def get_recalls_precisions(conf_matrix):

    precisions = np.diagonal(conf_matrix) / np.sum(conf_matrix, axis=1)
    recalls = np.diagonal(conf_matrix) / np.sum(conf_matrix, axis=0)

    return precisions, recalls

"""def avg_classification_rate(confusion_matrix):
    true_pred = 0
    for i in range(len(confusion_matrix)):
        true_pred += confusion_matrix[i][i]
    return true_pred/np.sum(confusion_matrix)"""

def get_f1_scores(conf_matrix):
    precisions, recalls = get_recalls_precisions(conf_matrix)

    return 2 * (precisions * recalls) / (precisions + recalls)


def get_accuracy(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)

    return np.sum(np.diagonal(conf_matrix)) / np.sum(conf_matrix)


def get_metrics(y_true, y_pred, printout=False):
    y_pred=[int(i) for i in y_pred]
    y_true=[int(i) for i in y_true]
    conf_matrix = confusion_matrix(y_true, y_pred)
    precision, recall = get_recalls_precisions(conf_matrix)
    f1 = get_f1_scores(conf_matrix)
    accuracy = get_accuracy(y_true, y_pred)

    if printout:
        print('---RESULT METRICS---')
        print('Precisions:  ', precision)
        print('Recalls:     ', recall)
        print('F1 Score:    ', f1)
        print('Avg Accuracy:', accuracy)

    return precision, recall, f1, accuracy


