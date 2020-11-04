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
        print('\nA detailed explanation of the code can be found in our readme file\n')
        while True:
            path = input('\nInput path to dataset\n')
            try:
                data = np.loadtxt(path)
                break
            except:
                print('Path not valid! Try again')

        print('Please select model\n')
        model = input('1 for DT, 2 for DT + pruning, 3 for CV + DT, 4 for CV + DT + pruning, 5 for depth evaluation\n')
        # cases for each model
        if model == '1':
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
            cm=ev.confusion_matrix(test[:,-1],y_pred)
            i=ev.get_metrics(cm,printout=True)
            print('To continue, you may need to close the plot windows first')
            ev.plot_conf_matrix(cm)
            print('Visualising the pruned trees')            
            model.visualise_tree()

            input('\nTo restart, hit enter\n')

        if model == '2':
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
            
            cm=ev.confusion_matrix(test[:,-1],y_pred)
            i=ev.get_metrics(cm,printout=True)    
            print('To continue, you may need to close the plot windows first')            
            ev.plot_conf_matrix(cm,title='Unpruned')
            print('Visualising the unpruned trees')            
            model.visualise_tree()            

            print('\nPruning...\n')

            model.prune_tree(validation)
            print('Max depth of tree after pruning:', model.get_max_depth())
            y_pred = model.predict(test[:, :-1])

            # evaluate
            cm=ev.confusion_matrix(test[:,-1],y_pred)
            i=ev.get_metrics(cm,printout=True)    
            print('To continue, you may need to close the plot windows first')            
            ev.plot_conf_matrix(cm,title='Pruned')
            print('Visualising the pruned trees')
            model.visualise_tree()            

            input('\nTo restart, hit enter\n')

        if model == '3':
            print('Training 10 fold CV...')
            results = cv.grow_binary_trees(data)
            print('Results')
            print('-' * 53 + '\n')
            r=ev.print_results(results)
            for i in r.keys():
                print(i)
                print(np.round(r[i],4))            
            input('\nTo restart, hit enter\n')

        if model == '4':
            print('Training 10 fold CV and then pruning it...')
            results, results_pruned = cv.grow_binary_trees(data, pruning=True)
            print('\nUnpruned Results')
            print('-' * 53 + '\n')
            r=ev.print_results(results)
            for i in r.keys():
                print(i)
                print(np.round(r[i],4)) 
            print('\nPruned Results')
            print('-' * 53 + '\n')
            r=ev.print_results(results_pruned)
            for i in r.keys():
                print(i)
                print(np.round(r[i],4)) 
            input('\nTo restart, hit enter\n')

        if model == '5':
            print('Training...\n')
            print('To continue later, you may need to close the plot window first')
            dp.plot_both(data)
            print('Training complete!\nFor a more detailed explanation of these graphs, please refer to our report!')
            input('\nTo restart, hit enter\n')
