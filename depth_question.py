import numpy as np
import find_split as fs
import evaluation as ev
import matplotlib.pyplot as plt
from trees import binarySearchTree

def get_values(data):
    """
    For each given depth, calculate the maximum depth
    and accuracies of the unpruned and pruned tree
    """
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
        cm=ev.confusion_matrix(y_true, y_pred)
        accuracies.append(ev.get_class_rate(cm))

        model.prune_tree(validation)
        depth2 = model.get_max_depth()
        y_pred2 = [x.item() for x in model.predict(test[:, :-1])]
        y_true2 = [int(val.item()) for val in test[:, -1]]
        pruned_depths.append(depth2)
        cm2=ev.confusion_matrix(y_true2, y_pred2)
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
    
def plot_both(data):
    """
    Creates plots measuring the:
    1) relationship between unpruned and pruned tree depths
    2) relationship between accuracy and tree depths for both trees
    """
    depths, accuracies, pruned_depths, pruned_accuracies = get_values(data)
    plt.figure(figsize=[12,6])
    plt.subplot(1,2,1)
    plt.plot(np.arange(len(depths)), np.arange(len(depths)), alpha=0.2, color='blue')
    plt.scatter(depths, pruned_depths)
    plt.ylabel('Tree depths after pruning')
    plt.xlabel('Tree depths before pruning')
    plt.subplot(1,2,2)
    plt.plot(depths, accuracies, color='b', label='Unpruned tree')
    plt.plot(pruned_depths, pruned_accuracies, color='orange', label='Pruned tree')
    plt.ylabel('Accuracy')
    plt.xlabel('Tree depth')
    plt.legend()
    plt.show()
    return
