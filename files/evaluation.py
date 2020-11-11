import numpy as np
import matplotlib.pyplot as plt

y_true = [1, 2, 3, 4, 1, 2, 3, 4]
y_pred = [1, 1, 3, 4, 1, 2, 2, 4]

def confusion_matrix(y_true, y_pred, normalised=True):

    y_true=[int(i) for i in y_true]
    y_pred=[int(i) for i in y_pred]    

    ret_matrix = np.zeros((4, 4))
    for x, y in zip(y_true, y_pred):
        ret_matrix[x - 1][y - 1] += 1

    if normalised:
        return ret_matrix / np.sum(ret_matrix, axis=1).reshape(-1, 1)
    else:
        return ret_matrix

def get_recalls_precisions(conf_matrix):

    precisions = np.diagonal(conf_matrix) / np.sum(conf_matrix, axis=0)
    recalls = np.diagonal(conf_matrix) / np.sum(conf_matrix, axis=1)

    return precisions, recalls


def get_f1_scores(conf_matrix):
    precisions, recalls = get_recalls_precisions(conf_matrix)

    return 2 * (precisions * recalls) / (precisions + recalls)

def get_class_rate(conf_matrix):

    return np.sum(np.diagonal(conf_matrix)) / np.sum(conf_matrix)


def get_metrics(conf_matrix,printout=False):

    precision, recall = get_recalls_precisions(conf_matrix)
    f1 = get_f1_scores(conf_matrix)
    class_rate = get_class_rate(conf_matrix)
    if printout:
        print('---RESULT METRICS---')
        print('Precisions:  ', precision)
        print('Recalls:     ', recall)
        print('F1 Score:    ', f1)
        print('Avg Accuracy:', class_rate)

    return precision, recall, f1, class_rate


def get_averages(results):
    '''
    Returns average metrics from ten folds + half of confidence interval width for each metric.
    '''
    precisions = []
    recalls = []
    f1_scores = []
    class_rates = []
    conf_matrix = np.zeros((4, 4))
    for i in range(len(results)):
        prec, rec, f1, c_r = get_metrics(results[i])

        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)
        class_rates.append(c_r)
        conf_matrix += results[i]

    return {
        'precision': np.mean(np.array(precisions), axis=0),
        'precision_CI': np.std(np.array(precisions), ddof=1, axis=0) * 1.96 / np.sqrt(10),
        'recall': np.mean(np.array(recalls), axis=0),
        'recall_CI': np.std(np.array(recalls), ddof=1, axis=0) * 1.96 / np.sqrt(10),
        'f1_score': np.mean(np.array(f1_scores), axis=0),
        'f1_score_CI': np.std(np.array(f1_scores), ddof=1, axis=0) * 1.96 / np.sqrt(10),
        'avg_class_rate': np.mean(np.array(class_rates)),
        'avg_class_rate_CI': np.std(np.array(class_rates), ddof=1) * 1.96 / np.sqrt(10),
        'avg_conf_matrix': conf_matrix / 10
    }


def print_results(results):
    '''
    Converts get_averages() results to make it suitable for printing out
    '''
    metrics = get_averages(results)
    return {
        'Precision': metrics['precision'],
        'Precision 95% CI': list(zip(metrics['precision'] - metrics['precision_CI'],
                                     metrics['precision'] + metrics['precision_CI'])),
        'Recall': metrics['recall'],
        'Recall 95% CI': list(zip(metrics['recall'] - metrics['recall_CI'],
                                  metrics['recall'] + metrics['recall_CI'])),
        'F1 Score': metrics['recall'],
        'F1 Score 95% CI': list(zip(metrics['f1_score'] - metrics['f1_score_CI'],
                                    metrics['f1_score'] + metrics['f1_score_CI'])),
        'Avg. Class. Rate': metrics['avg_class_rate'],
        'Avg. Class. Rate 95% CI': (metrics['avg_class_rate'] - metrics['avg_class_rate_CI'],
                                    metrics['avg_class_rate'] + metrics['avg_class_rate_CI'])
    }


def plot_conf_matrix(confusion_matrix, title=None):
    fig, ax = plt.subplots(1, 1)
    plt.imshow(confusion_matrix)
    plt.xlabel("predicted")
    plt.ylabel("true")
    if title:
        plt.title(title)
    label_list = ['1', '2', '3', '4']

    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks([0, 1, 2, 3])

    ax.set_xticklabels(label_list)
    ax.set_yticklabels(label_list)

    for i in range(4):
        for j in range(4):
            color = 'black' if i == j else 'w'
            text = ax.text(i, j, np.round(confusion_matrix[i][j], 4),
                           ha="center", va="center", color=color)
    plt.show()
