import numpy as np
import evaluation
from trees import binarySearchTree


def grow_binary_trees(data, stratified=False, pruning=False):
    '''
    Splits the data set into 10 folds and uses each fold as a testing set once.

    If pruning: returns two dictionaries of results, for pruned and unpruned trees
    '''

    # Create 10 Folds: Group indices into 10 folds
    np.random.shuffle(data)
    data = data.reshape((10, -1, 8))

    if pruning:
        results, results_pruned = {}, {}
        for i, fold in enumerate(data):
            results[i] = {}
            results_pruned[i] = {}

            test_data = fold
            train_val_data = np.delete(data, i, axis=0)
            pruning_data, train_data = train_val_data[0], np.vstack(train_val_data[1:])

            tree = binarySearchTree(train_data)

            results[i]['y_true'] = test_data[:, -1].astype(int)
            results[i]['y_pred'] = tree.predict(test_data)

            tree.prune_tree(pruning_data)

            results_pruned[i]['y_true'] = test_data[:, -1].astype(int)
            results_pruned[i]['y_pred'] = tree.predict(test_data)

        return results, results_pruned

    else:
        results = {}
        for i, fold in enumerate(data):
            results[i] = dict()

            test_data = fold
            train_data = np.vstack(np.delete(data, i, axis=0))

            tree = binarySearchTree(train_data)

            results[i]['y_true'] = test_data[:, -1].astype(int)
            results[i]['y_pred'] = tree.predict(test_data)

        return results


def get_confusion_matrix(y_true, y_pred):
    ret_matrix = np.zeros((4, 4))
    for x, y in zip(y_pred, y_true):
        ret_matrix[x - 1][y - 1] += 1

    return ret_matrix

def get_recalls_precisions(conf_matrix):

    precisions = np.diagonal(conf_matrix) / np.sum(conf_matrix, axis=1)
    recalls = np.diagonal(conf_matrix) / np.sum(conf_matrix, axis=0)

    return precisions, recalls


def get_f1_scores(conf_matrix):
    precisions, recalls = get_recalls_precisions(conf_matrix)

    return 2 * (precisions * recalls) / (precisions + recalls)

def get_class_rate(conf_matrix):

    return np.sum(np.diagonal(conf_matrix)) / np.sum(conf_matrix)

def get_metrics(conf_matrix):

    precision, recall = get_recalls_precisions(conf_matrix)
    f1 = get_f1_scores(conf_matrix)
    class_rate = get_class_rate(conf_matrix)

    return precision, recall, f1, class_rate


def get_averages(results):
    precisions = []
    recalls = []
    f1_scores = []
    class_rates = []
    for i in range(len(results)):
        conf_matrix = get_confusion_matrix(results[i]['y_true'], results[i]['y_pred'])
        prec, rec, f1, c_r = get_metrics(conf_matrix)

        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)
        class_rates.append(c_r)
    return {
        'precision': np.mean(np.array(precisions), axis=0),
        'recall': np.mean(np.array(recalls), axis=0),
        'f1_score': np.mean(np.array(f1_scores), axis=0),
        'avg_class_rate': np.mean(np.array(class_rates))
    }
