import numpy as np
import find_split as fs
import evaluation as ev
import matplotlib.pyplot as plt
from trees import binarySearchTree

def get_values(data):
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
        accuracies.append(ev.get_accuracy(y_true, y_pred))

        model.prune_tree(validation)
        depth2 = model.get_max_depth()
        y_pred2 = [x.item() for x in model.predict(test[:, :-1])]
        y_true2 = [int(val.item()) for val in test[:, -1]]
        pruned_depths.append(depth2)
        pruned_accuracies.append(ev.get_accuracy(y_true2, y_pred2))

    print(depths, accuracies)
    print(pruned_depths, pruned_accuracies)
    return depths, accuracies, pruned_depths, pruned_accuracies

def graph_depths(data):
    depths, _, pruned_depths, _ = get_values(data)
    plt.plot(np.arange(len(depths)), np.arange(len(depths)), alpha=0.2, color='blue')
    plt.scatter(depths, pruned_depths)
    plt.ylabel('Tree depths after pruning')
    plt.xlabel('Tree depths before pruning')
    plt.show()

def graph_depth_accuracy(data):
    depths, accuracies, pruned_depths, pruned_accuracies = get_values(data)
    plt.plot(depths, accuracies, color='b', label='unpruned tree')
    plt.plot(pruned_depths, pruned_accuracies, color='orange', label='pruned tree')
    plt.ylabel('accuracy')
    plt.xlabel('tree depth')
    plt.legend()
    plt.show()

noisy_data = np.loadtxt('noisy_dataset.txt')
clean_data = np.loadtxt('clean_dataset.txt')
noisy_data=noisy_data.copy()
clean_data = clean_data.copy()

#graph_depths(noisy_data)
graph_depth_accuracy(clean_data)
