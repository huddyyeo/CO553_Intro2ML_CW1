import numpy as np
import evaluation
from trees import binarySearchTree


def grow_binary_trees(data, stratified=False, pruning=False):
    '''
    Splits the data set into 10 folds and uses each fold as a testing set once.
    Calculates metrics for each fold to get standard errors.
    Data split - stratified or fully random. #TODO
    '''

    # Create 10 Folds: Group indices into 10 folds
    np.random.shuffle(data)
    data = data.reshape((10, -1, 8))

    # Training folds
    results = {}

    if pruning:
        for i, fold in enumerate(data):
            results[i] = dict()

            test_data = fold
            train_val_data = np.delete(data, i, axis=0)
            print(train_val_data.shape)
            np.random.shuffle(train_val_data)
            pruning_data, train_data = train_val_data[0], np.vstack(train_val_data[1:])

            print(test_data.shape, train_data.shape, pruning_data.shape)

            tree = binarySearchTree(train_data)

            results[i]['y_true'] = test_data[:, -1].astype(int)
            results[i]['y_pred_unpruned'] = tree.predict(test_data)

            tree.prune_tree(pruning_data)

            results[i]['y_pred_pruned'] = tree.predict(test_data)

    else:

        for i, fold_ids in enumerate(fold_indices):
            results[i] = dict()
            train_ids = np.delete(np.arange(data.shape[0]), fold_ids)
            test_data, test_labels = data[fold_ids][:, :-1], data[fold_ids][:, -1]
            train_data = data[train_ids]

            tree = binarySearchTree(train_data)

            results[i]['index'] = fold_ids
            results[i]['y_true'] = test_labels.astype(int)
            results[i]['y_pred'] = tree.predict(test_data)

    return results


def get_confusion_matrix(y_true, y_pred):
    ret_matrix = np.zeros((4, 4))
    for x, y in zip(y_pred, y_true):
        ret_matrix[x - 1][y - 1] += 1

    return ret_matrix


def get_recalls_precisions(y_true, y_pred):
    # precision = diagonal / row
    # recall = diagonal / column
    conf_matrix = get_confusion_matrix(y_true, y_pred)

    precisions = np.diagonal(conf_matrix) / np.sum(conf_matrix, axis=1)
    recalls = np.diagonal(conf_matrix) / np.sum(conf_matrix, axis=0)

    return precisions, recalls


def get_f1_scores(y_true, y_pred):
    precisions, recalls = get_recalls_precisions(y_true, y_pred)

    return 2 * (precisions * recalls) / (precisions + recalls)


def get_accuracy(y_true, y_pred):
    conf_matrix = get_confusion_matrix(y_true, y_pred)

    return np.sum(np.diagonal(conf_matrix)) / np.sum(conf_matrix)

def get_metrics(y_true, y_pred, printout=False):

    precision, recall = get_recalls_precisions(y_true, y_pred)
    f1 = get_f1_scores(y_true, y_pred)
    accuracy = get_accuracy(y_true, y_pred)

    if printout:
        print('---RESULT METRICS---')
        print('Precisions:  ', precision)
        print('Recalls:     ', recall)
        print('F1 Score:    ', f1)
        print('Avg Accuracy:', accuracy)

    return precision, recall, f1, accuracy


class ResultPlotter:
    def __init__(self, results):
        self.results = results

        recalls, precisions, f1_scores, accuracies = self.get_folds_metrics()
        self.recalls = np.array(recalls)
        self.precisions = np.array(precisions)
        self.f1_scores = np.array(f1_scores)
        self.accuracies = np.array(accuracies)

    def get_folds_metrics(self):
        recalls = []
        precisions = []
        f1_scores = []
        accuracies = []

        for i in range(10):
            y_true, y_pred = self.results[i]['y_true'], self.results[i]['y_pred']
            precision, recall, f1, accuracy = get_metrics(y_true, y_pred)

            recalls.append(recall)
            precisions.append(precision)
            f1_scores.append(f1)
            accuracies.append(accuracy)

        return recalls, precisions, f1_scores, accuracies

    def recall_plot(self):
        return self.recalls

    def precision_plot(self):
        pass

    def f1_plot(self):
        pass

    def accuracy_plot(self):
        pass
