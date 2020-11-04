import argparse
import numpy as np
import find_split as fs
import evaluation as ev
import matplotlib.pyplot as plt
from trees import binarySearchTree
import depth_question as dp
import cross_validation as cv
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    while True:
        print('-' * 53)
        print('Decision Tree training code for Intro2ML coursework 1')
        print('-' * 53)
        while True:
            path = input('\nInput path to dataset\n')
            try:
                data = np.loadtxt(path)
                break
            except:
                print('Path not valid! Try again')

        print('Please select model\n')
        model = input('0 for DT, 1 for DT + pruning, 2 CV + DT, 3 CV + DT + pruning, 4 depth question\n')
        # cases for each model
        if model == '0':
            while True:
                split = float(input('Enter training data split value, eg 0.7\n'))
                if split < 0 or split > 1:
                    print('Invalid split entered')
                else:
                    break
            limit = input('Please enter a depth limit for the decision tree, if you do not want a limit just press enter\n')
            if limit == '':
                print('No limit entered')
                limit = None
            else:
                limit = int(limit)

            np.random.shuffle(data)
            train = data[:int(len(data) * split)]
            test = data[int(len(data) * split):]

            model = binarySearchTree(train, limit=limit)
            print('Max depth of tree is', model.get_max_depth())

            y_pred = model.predict(test[:, :-1])
            ev.get_metrics(test[:, -1], y_pred, printout=True)
            print('To continue, you may need to close the plot windows first')
            ev.confusion_matrix(test[:, -1], y_pred, plot=True)
            print('Visualising the pruned trees')            
            model.visualise_tree()

            input('\nTo restart, hit enter\n')

        if model == '1':
            split = float(input('Enter training data split value, eg 0.7\n'))
            while True:
                if split < 0 or split > 1:
                    print('Invalid split entered')
                else:
                    break
            print('You have entered ' + str(split) + '\n')
            print('A validation set of ' + str(round((1 - split) / 2, 2)) + ' will be used\n')
            limit = input('Please enter a depth limit for the decision tree, if you do not want a limit just press enter\n')
            if limit == '':
                print('No limit entered')
                limit = None
            else:
                limit = int(limit)
            np.random.shuffle(data)
            t_split = split
            v_split = (1 + split) / 2
            train = data[:int(len(data) * t_split)]
            validation = data[int(len(data) * t_split):(int(len(data) * v_split))]
            test = data[int(len(data) * v_split):]

            model = binarySearchTree(train, limit=limit)
            print('Max depth of tree before pruning:', model.get_max_depth())
            y_pred = model.predict(test[:, :-1])
            # evaluate
            ev.get_metrics(test[:, -1], y_pred, printout=True)
            print('To continue, you may need to close the plot windows first')
            ev.confusion_matrix(test[:, -1], y_pred, plot=True, title='Unpruned')
            print('Visualising the pruned trees')            
            model.visualise_tree()            

            print('\nPruning...\n')

            model.prune_tree(validation)
            print('Max depth of tree after pruning:', model.get_max_depth())
            y_pred = model.predict(test[:, :-1])

            # evaluate
            ev.get_metrics(test[:, -1], y_pred, printout=True)
            print('To continue, you may need to close the plot window first')
            ev.confusion_matrix(test[:, -1], y_pred, plot=True, title='Pruned')
            print('Visualising the pruned trees')
            model.visualise_tree()            

            input('\nTo restart, hit enter\n')

        if model == '2':
            print('Training 10 fold CV...')
            results = cv.grow_binary_trees(data)
            results = cv.print_results(results)
            print('Results')
            print('-' * 53 + '\n')
            for i in list(results.keys()):
                print(i, results[i])
            input('\nTo restart, hit enter\n')

        if model == '3':
            print('Training 10 fold CV and then pruning it...')
            results, results_pruned = cv.grow_binary_trees(data, pruning=True)
            results = cv.print_results(results)
            results_pruned = cv.get_averages(results_pruned)
            print('\nUnpruned Results')
            print('-' * 53 + '\n')
            for i in list(results.keys()):
                print(i, results[i])
            print('\nPruned Results')
            print('-' * 53 + '\n')
            for i in list(results_pruned.keys()):
                print(i, results_pruned[i])
            input('\nTo restart, hit enter\n')

        if model == '4':
            print('Training...\n')
            print('To continue later, you may need to close the plot window first')
            dp.plot_both(data)
            print('Training complete!\nFor a more detailed explanation of these graphs, please refer to our report!')
            input('\nTo restart, hit enter\n')
