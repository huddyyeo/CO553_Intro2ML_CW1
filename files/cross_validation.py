import numpy as np
import evaluation as ev
from trees import binarySearchTree


def grow_binary_trees(data_input, stratified=False, pruning=False):
    '''
    Splits the data set into 10 folds and uses each fold as a testing set once.
    Calculates metrics for each fold to get standard errors.
    Data split - stratified or fully random. #TODO
    '''

    # Create 10 Folds: Group indices into 10 folds
    data = data_input.copy()
    np.random.shuffle(data)
    data = data.reshape((10, -1, 8))

    if pruning:
        results, results_pruned = {}, {}
        for i, test_fold in enumerate(data):
            results[i] = np.zeros((4, 4))
            results_pruned[i] = np.zeros((4, 4))

            train_val_data = np.delete(data, i, axis=0)

            for j, val_fold in enumerate(train_val_data):

                train_data = np.vstack(np.delete(train_val_data, j, axis=0))

                tree = binarySearchTree(train_data)

                results[i] += ev.confusion_matrix(test_fold[:, -1].astype(int),
                                                  tree.predict(test_fold),
                                                  normalised=True)

                tree.prune_tree(val_fold)

                results_pruned[i] += ev.confusion_matrix(test_fold[:, -1].astype(int),
                                                         tree.predict(test_fold),
                                                         normalised=True)
            results[i] /= 9
            results_pruned[i] /= 9

        return results, results_pruned

    else:
        results = {}
        for i, test_fold in enumerate(data):

            train_data = np.vstack(np.delete(data, i, axis=0))

            tree = binarySearchTree(train_data)

            results[i] = ev.confusion_matrix(test_fold[:, -1].astype(int),
                                             tree.predict(test_fold),
                                             normalised=True)

        return results


def metrics_pruning_plot(results_clean, results_clean_pruned, results_noisy, results_noisy_pruned, savefile=False):
    '''
    Performs CV and pruning on the data sets and plots the resulting metrics
    '''

    # Get metrics and CIs
    metrics_clean = ev.get_averages(results_clean)
    metrics_clean_pruned = ev.get_averages(results_clean_pruned)
    metrics_noisy = ev.get_averages(results_noisy)
    metrics_noisy_pruned = ev.get_averages(results_noisy_pruned)

    x = np.arange(4)  # the label locations
    width = 0.35  # the width of the bars

    fig, axs = plt.subplots(figsize=(8, 9), ncols=2, nrows=3)
    for i, title in zip(range(3), ['precision', 'recall', 'f1_score']):
        # Clean data plots:
        axs[i][0].grid(True)
        axs[i][0].set_ylim(0.5, 1.1)
        rects1 = axs[i][0].bar(x=x - width / 2,
                               height=metrics_clean[title],
                               width=width,
                               yerr=metrics_clean[title + '_CI'],
                               label='Unpruned',
                               zorder=3)
        rects2 = axs[i][0].bar(x=x + width / 2,
                               height=metrics_clean_pruned[title],
                               width=width,
                               yerr=metrics_clean_pruned[title + '_CI'],
                               label='Pruned',
                               zorder=3)
        axs[i][0].set_xticks(np.arange(4))
        axs[i][0].set_xticklabels([])
        axs[i][0].set_ylabel(title.capitalize(), fontsize=13)

        # Noisy data plots:
        axs[i][1].grid(True)
        axs[i][1].set_ylim(0.5, 1.1)
        rects1 = axs[i][1].bar(x=x - width / 2,
                               height=metrics_noisy[title],
                               width=width,
                               yerr=metrics_noisy[title + '_CI'],
                               label='Unpruned',
                               zorder=3)
        rects2 = axs[i][1].bar(x=x + width / 2,
                               height=metrics_noisy_pruned[title],
                               width=width,
                               yerr=metrics_noisy_pruned[title + '_CI'],
                               label='Pruned',
                               zorder=3)
        axs[i][1].set_xticks(np.arange(4))
        axs[i][1].set_xticklabels([])
        axs[i][1].set_yticklabels([])
    axs[0][0].set_title('Clean Data', y=1.05, ha='center', fontsize=16)
    axs[0][1].set_title('Noisy Data', y=1.05, ha='center', fontsize=16)
    axs[1][1].legend(loc='upper right')
    axs[2][0].set_xticklabels([f'Room {i}' for i in range(1, 5)])
    axs[2][1].set_xticklabels([f'Room {i}' for i in range(1, 5)])
    plt.tight_layout(pad=0.5)
    if savefile:
        plt.savefig('pruned_unpruned.png', format='png')
