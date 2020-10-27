import numpy as np
import matplotlib.pyplot as plt

y_true = [1,2,3,4,1,2,3,4]
y_predicted = [1,1,3,4,1,2,2,4]

def confusion_matrix(y_true, y_predicted,plot=False):
    y_true=[int(i) for i in y_true]
    y_predicted=[int(i) for i in y_predicted]
    labels = np.unique(y_true)
    N = len(labels)

    confusion_matrix = [ [ 0 for i in range(N) ] for j in range(N) ] # 4 x 4 matrix

    for i,room in enumerate(y_true):
        confusion_matrix[room-1][y_predicted[i]-1] += 1
    if plot==True:
        
        plt.matshow(confusion_matrix)
        plt.xlabel("predicted")
        plt.ylabel("true")
        """plt.xlim(1,4) !!!! make room plots form 1 to 4
        plt.ylim(1,4)"""
        for i in range(N):
            for j in range(N):
                text = plt.text(i,j, confusion_matrix[i][j],
                               ha="center", va="center", color="w")
        plt.show() 
    return confusion_matrix



def avg_recall_precision(confusion_matrix):
    N = len(confusion_matrix)
    avg_true_pos = 0
    sum_false_neg = 0

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
