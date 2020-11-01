import numpy as np
import matplotlib.pyplot as plt

y_true = [1,2,3,4,1,2,3,4]
y_predicted = [1,1,3,4,1,2,2,4]


def confusion_matrix(y_true, y_pred,plot=False):
    confusion_matrix = np.zeros((4, 4))
    if type(y_true[0])!= int:
        y_true = [int(x) for x in y_true.tolist()]
    if type(y_pred[0])!= int:
        y_pred = [int(x) for x in train_pred.tolist()]
    for x, y in zip(y_pred, y_true):
        confusion_matrix[x - 1][y - 1] += 1


    if plot:
        fig, ax = plt.subplots(1,1)
        plt.imshow(confusion_matrix)
        plt.xlabel("predicted")
        plt.ylabel("true")

        label_list = ['1','2', '3', '4']

        ax.set_xticks([0,1,2,3])
        ax.set_yticks([0, 1, 2, 3])

        ax.set_xticklabels(label_list)
        ax.set_yticklabels(label_list)

        for i in range(4):
            for j in range(4):
                text = ax.text(i,j, confusion_matrix[i][j],
                               ha="center", va="center", color="w")
        fig.show()
    return confusion_matrix



def avg_recall_precision(confusion_matrix):
    N = len(confusion_matrix)

    recall = {}
    precision = {}

    for room in range(N):
        true_pos = confusion_matrix[room][room]
        recall[room] = true_pos / sum(confusion_matrix[:][room])
        precision[room] = true_pos / sum(confusion_matrix[room][:])
    #print(sum(recall.values())/N, sum(precision.values()) / N)
    return sum(recall.values())/N, sum(precision.values()) / N


avg_recall_precision(confusion_matrix(y_true, y_predicted))

def f1_score(recall, precision):
    return 2/((1/recall)+(1/precision))

def avg_classification_rate(confusion_matrix):
    true_pred = 0
    for i in range(len(confusion_matrix)):
        true_pred += confusion_matrix[i][i]
    return true_pred/np.sum(confusion_matrix)
