import numpy as np
import find_split as fs
import evaluation as ev
import matplotlib.pyplot as plt
from trees import binarySearchTree


def get_values(data_input):
    """
    For each given depth, calculate the maximum depth
    and accuracies of the unpruned and pruned tree
    """
    data = data_input.copy()
    np.random.shuffle(data)
    t_split = 0.8
    v_split = 0.9
    train = data[:int(len(data) * t_split)]
    validation = data[int(len(data) * t_split):(int(len(data) * v_split))]
    test = data[int(len(data) * v_split):]
    depths = []
    pruned_depths = []
    accuracies = []
    pruned_accuracies = []

    for i in range(1, 20):
        model = binarySearchTree(train, limit=i)
        depth = model.get_max_depth()
        y_pred = [x.item() for x in model.predict(test[:, :-1])]
        y_true = [int(val.item()) for val in test[:, -1]]
        depths.append(depth)
        cm = ev.confusion_matrix(y_true, y_pred)
        accuracies.append(ev.get_class_rate(cm))

        model.prune_tree(validation)
        depth2 = model.get_max_depth()
        y_pred2 = [x.item() for x in model.predict(test[:, :-1])]
        y_true2 = [int(val.item()) for val in test[:, -1]]
        pruned_depths.append(depth2)
        cm2 = ev.confusion_matrix(y_true2, y_pred2)
        pruned_accuracies.append(ev.get_class_rate(cm2))
    return depths, accuracies, pruned_depths, pruned_accuracies


def graph_depths(data):
    """
    Graphs the relationship between the depths of pre-pruned and pruned trees
    """
    depths, _, pruned_depths, _ = get_values(data)
    plt.plot(np.arange(len(depths)), np.arange(len(depths)), alpha=0.2, color='blue')
    plt.scatter(depths, pruned_depths)
    plt.ylabel('Tree depths after pruning')
    plt.xlabel('Tree depths before pruning')
    plt.show()


def graph_depth_accuracy(data):
    """
    Graph the relationship between the accuracy and the depth of the tree
    """
    depths, accuracies, pruned_depths, pruned_accuracies = get_values(data)
    plt.plot(depths, accuracies, color='b', label='unpruned tree')
    plt.plot(pruned_depths, pruned_accuracies, color='orange', label='pruned tree')
    plt.ylabel('accuracy')
    plt.xlabel('tree depth')
    plt.legend()
    plt.show()


def plot_both(data, title='default.png'):
    """
    Creates plots measuring the:
    1) relationship between unpruned and pruned tree depths
    2) relationship between accuracy and tree depths for both trees
    """
    depths, accuracies, pruned_depths, pruned_accuracies = get_values(data)
    pruned_accuracies = np.array(pruned_accuracies)
    error_pruned = 1.96 * (np.abs(pruned_accuracies - 1)) / np.sqrt(200)
    accuracies = np.array(accuracies)
    error = 1.96 * (np.abs(accuracies - 1)) / np.sqrt(200)

    fig, ax = plt.subplots(figsize=(12, 4), ncols=2)
    ax[0].plot(np.arange(max(depths) + 1), np.arange(max(depths) + 1), alpha=0.2, color='blue')
    ax[0].scatter(depths, pruned_depths)
    ax[0].set_ylabel('Tree depths after pruning')
    ax[0].set_xlabel('Tree depths before pruning')
    ax[0].set_xticks(np.arange(0, max(depths) + 1))

    ax[1].plot(depths, accuracies, color='b', label='Unpruned tree')
    ax[1].plot(pruned_depths, pruned_accuracies, color='orange', label='Pruned tree')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_xlabel('Tree depth')
    ax[1].fill_between(depths, accuracies - error, accuracies + error, alpha=0.2, color='b', zorder=1)
    ax[1].fill_between(pruned_depths, pruned_accuracies - error_pruned, pruned_accuracies + error_pruned, alpha=0.2, color='orange', zorder=1)

    ax[1].set_xticks(np.arange(0, max(depths) + 1))
    ax[1].legend()
    plt.tight_layout()
    plt.show()
