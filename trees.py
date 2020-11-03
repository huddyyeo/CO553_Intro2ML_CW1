import numpy as np
import find_split as fs
import evaluation as ev
import matplotlib.pyplot as plt


class binarySearchTree:
    def __init__(self, data, depth=-1, label=None, limit=None):
        self.left_child = None
        self.right_child = None
        self.depth = depth + 1
        self.label = label
        self.split_value = None
        self.split_router = None

        # default value none, 0 for choosing not to prune, 1 for testing during pruning, 2 for permanently pruned
        self.prune = None

        # if we have passed the limit, set this to be a leaf node
        if limit is not None:
            if self.depth >= limit:
                self.label = np.argmax(np.bincount([int(i) for i in data[:, -1]]))
                return

        # set future prune value to most common label in data
        self.prune_label = np.argmax(np.bincount([int(i) for i in data[:, -1]]))

        if len(np.unique(data[:, -1])) == 1:  # assuming last column is for labels
            self.label = data[0, -1]

        else:
            # not all samples have same label, do a split
            # split router: router from [1,...,6]
            # split_value: router value
            split_router, split_value, temp_data = fs.find_split(data)
            self.split_value = split_value
            self.split_router = split_router
            l_data = temp_data[0]
            r_data = temp_data[1]

            # recursively search the tree, branching into 2
            self.left_child = binarySearchTree(l_data, self.depth, limit=limit)
            self.right_child = binarySearchTree(r_data, self.depth, limit=limit)

    def get_max_depth(self):
        # search each branch recursively and get max depth
        max_depth = [self.depth]
        if self.left_child:
            max_depth.append(self.left_child.get_max_depth())
        if self.right_child:
            max_depth.append(self.right_child.get_max_depth())
        return max(max_depth)

    def prune_1_node(self, current_path=['parent']):
        if self.label:
            return

        # prune refers to whether or not we have tested it before
        if self.prune != None:
            # if this node is already to be tested for pruning, we cannot be searching for any other nodes to prune!
            if self.prune == 1:

                raise ValueError('tried to prune two nodes')

            # else this means we have already pruned or chosen not to prune this branch, return True
            return True

        # if either child is a leaf or already pruned,
        if self.left_child.label and self.right_child.label:
            #print('set 1 to prune at path',current_path)
            self.prune = 1
            # set this node for pruning
            return False

        else:
            # recursively search for a prunable tree
            # if False returned, we found a node to prune
            # if True returned, we have pruned all possible nodes
            l_path = current_path + ['l']
            l = self.left_child.prune_1_node(current_path=l_path)
            if l == False:
                return False
            r_path = current_path + ['r']
            r = self.right_child.prune_1_node(current_path=r_path)
            if r == False:
                return False

        return True

    def get_f1(self, data):
        pred = self.predict(data[:, :-1])
        cm = ev.confusion_matrix(data[:, -1], pred)
        return ev.get_f1_scores(cm)

    def set_prune_status(self, pruned=False):

        if self.label:
            return
        if self.prune:

            if self.prune == 0 or self.prune == 2:
                return
            if pruned == False:
                self.prune = 0
                return
            else:
                self.prune = 2
                self.label = self.prune_label
                self.left_child = None
                self.right_child = None
                return
        self.left_child.set_prune_status(pruned)
        self.right_child.set_prune_status(pruned)

    def prune_tree(self, data, print_path=False):
        end = False

        val_error = np.mean(self.get_f1(data))
        while (not end):
            # find node to prune
            end = self.prune_1_node()
            if print_path == True:
                print('current f1:', val_error)
            # validate and get error
            new_f1 = np.mean(self.get_f1(data))
            if print_path == True:
                print('new f1 score:', new_f1)

            # if error is better, prune or else dont prune
            if new_f1 >= val_error:
                if print_path == True:
                    print('pruned 1!')
                self.set_prune_status(pruned=True)
                val_error = new_f1
            else:
                self.set_prune_status(pruned=False)
                if print_path == True:
                    print('did not prune')

    def predict_one(self, data):
        if self.prune:
            if self.prune > 0:
                return np.array([self.prune_label])
        if self.label:
            return np.array([int(self.label)])
        else:
            if data[self.split_router - 1] <= self.split_value:
                return self.left_child.predict_one(data)

            else:
                return self.right_child.predict_one(data)

    def predict(self, data):

        data = np.squeeze(data)
        if len(data.shape) > 1:
            return np.array([self.predict_one(i) for i in data]).flatten()
        else:
            return self.predict_one(data).flatten()

    def get_paths(self, path=['root'], objects=[]):
        if self.label != None:
            objects.append((path, f'Room {int(self.label)}'))
        else:
            objects.append((path, f'Router {self.split_router}\nX = {self.split_value}'))
            l_path = path + ['l']
            r_path = path + ['r']
            self.left_child.get_paths(l_path, objects)
            self.right_child.get_paths(r_path, objects)
        return objects

    def visualise_tree(self):
        paths = self.get_paths()
        bbox_node = {'boxstyle': "round", 'ec': 'black', 'fc': 'lightgrey'}
        bbox_label = {'boxstyle': "round", 'ec': 'black', 'fc': 'lightblue'}
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for pair in paths:
            path, label = pair
            n = len(path)
            if n > 4:
                continue
            x = 0.5
            y = 0.90
            for i, side in enumerate(path):

                if side == 'l':
                    x -= 0.25 / i
                if side == 'r':
                    x += 0.25 / i
                if side != 'root':
                    y -= 0.25

            if label[2] != 'o':
                ax.text(x, y, s=label, ha='center', fontsize=16 - (1.5 * len(path)), bbox=bbox_node)
                n = len(path)
                ax.arrow(x, y, -0.25 / n, -0.25)
                ax.arrow(x, y, 0.25 / n, -0.25)
            else:
                ax.text(x, y, s=label, ha='center', fontsize=16 - (1.5 * len(path)), bbox=bbox_label)
        plt.axis('off')
        plt.show()
