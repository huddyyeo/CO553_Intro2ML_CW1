import numpy as np
import evaluation
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

                results[i] += get_confusion_matrix(test_fold[:, -1].astype(int),
                                                   tree.predict(test_fold),
                                                   normalised=True)

                tree.prune_tree(val_fold)

                results_pruned[i] += get_confusion_matrix(test_fold[:, -1].astype(int),
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

            results[i] = get_confusion_matrix(test_fold[:, -1].astype(int),
                                              tree.predict(test_fold),
                                              normalised=True)

        return results


# def get_confusion_matrix(y_true, y_pred, normalised=True):
#     ret_matrix = np.zeros((4, 4))
#     for x, y in zip(y_true, y_pred):
#         ret_matrix[x - 1][y - 1] += 1

#     if normalised:
#         return ret_matrix / np.sum(ret_matrix, axis=1).reshape(-1, 1)
#     else:
#         return ret_matrix

# def get_recalls_precisions(conf_matrix):

#     precisions = np.diagonal(conf_matrix) / np.sum(conf_matrix, axis=0)
#     recalls = np.diagonal(conf_matrix) / np.sum(conf_matrix, axis=1)

#     return precisions, recalls


# def get_f1_scores(conf_matrix):
#     precisions, recalls = get_recalls_precisions(conf_matrix)

#     return 2 * (precisions * recalls) / (precisions + recalls)

# def get_class_rate(conf_matrix):

#     return np.sum(np.diagonal(conf_matrix)) / np.sum(conf_matrix)

# def get_metrics(conf_matrix):

#     precision, recall = get_recalls_precisions(conf_matrix)
#     f1 = get_f1_scores(conf_matrix)
#     class_rate = get_class_rate(conf_matrix)

#     return precision, recall, f1, class_rate


# def get_averages(results):
#     '''
#     Returns average metrics from ten folds + half of confidence interval width for each metric.
#     '''

#     precisions = []
#     recalls = []
#     f1_scores = []
#     class_rates = []
#     for i in range(len(results)):
#         prec, rec, f1, c_r = get_metrics(results[i])

#         precisions.append(prec)
#         recalls.append(rec)
#         f1_scores.append(f1)
#         class_rates.append(c_r)
#     return {
#         'precision': np.mean(np.array(precisions), axis=0),
#         'precision_CI': np.std(np.array(precisions), ddof=1, axis=0) * 1.96 / np.sqrt(10),
#         'recall': np.mean(np.array(recalls), axis=0),
#         'recall_CI': np.std(np.array(recalls), ddof=1, axis=0) * 1.96 / np.sqrt(10),
#         'f1_score': np.mean(np.array(f1_scores), axis=0),
#         'f1_score_CI': np.std(np.array(f1_scores), ddof=1, axis=0) * 1.96 / np.sqrt(10),
#         'avg_class_rate': np.mean(np.array(class_rates)),
#         'avg_class_rate_CI': np.std(np.array(class_rates), ddof=1) * 1.96 / np.sqrt(10)
#     }


# def print_results(results):
#     metrics = get_averages(results)
#     return {
#         'Precision': metrics['precision'],
#         'Precision 95% CI': list(zip(metrics['precision'] - metrics['precision_CI'],
#                                      metrics['precision'] + metrics['precision_CI'])),
#         'Recall': metrics['recall'],
#         'Recall 95% CI': list(zip(metrics['recall'] - metrics['recall_CI'],
#                                   metrics['recall'] + metrics['recall_CI'])),
#         'F1 Score': metrics['recall'],
#         'F1 Score 95% CI': list(zip(metrics['f1_score'] - metrics['f1_score_CI'],
#                                     metrics['f1_score'] + metrics['f1_score_CI'])),
#         'Avg. Class. Rate': metrics['avg_class_rate'],
#         'Avg. Class. Rate 95% CI': (metrics['avg_class_rate'] - metrics['avg_class_rate_CI'],
#                                     metrics['avg_class_rate'] + metrics['avg_class_rate_CI'])
#     }


def metrics_pruning_plot(results_clean, results_clean_pruned, results_noisy, results_noisy_pruned, savefile=False):
    '''
    Performs CV and pruning on the data sets and plots the resulting metrics
    '''

    # Get metrics and CIs
    metrics_clean = get_averages(results_clean)
    metrics_clean_pruned = get_averages(results_clean_pruned)
    metrics_noisy = get_averages(results_noisy)
    metrics_noisy_pruned = get_averages(results_noisy_pruned)

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
