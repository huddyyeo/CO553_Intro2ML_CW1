import numpy as np
import evaluation
from trees import binarySearchTree


def grow_binary_trees(data, stratified=False, pruning=None):
    '''
    Splits the data set into 10 folds and uses each fold as a testing set once.
    Calculates metrics for each fold to get standard errors.
    Data split - stratified or fully random. #TODO
    ----------
    :return: dict of dicts, {indices, true_labels, predicted_labels} for each fold.
    '''

    # Create 10 Folds: Group indices into 10 folds
    if stratified:
        pass
    else:
        fold_indices = np.random.permutation(np.arange(data.shape[0])).reshape(10, -1)

    # Training folds
    results = {'strtified': stratified}

    for i, gen in enumerate(fold_indices):
        results[i] = dict()
        train_mask = np.delete(np.arange(data.shape[0]), gen)
        test_data, test_labels = data[gen][:, :-1], data[gen][:, -1]
        train_data = data[train_mask]

        tree = binarySearchTree(train_data)

        results[i]['index'] = gen
        results[i]['true_y'] = test_labels
        results[i]['pred_y'] = tree.predict(test_data)

    return results
