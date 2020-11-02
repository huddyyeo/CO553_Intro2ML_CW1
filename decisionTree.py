import argparse
import numpy as np
import find_split as fs
import evaluation as ev
import matplotlib.pyplot as plt
from trees import binarySearchTree
import depth_question as dp
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    while True:
        print('-'*53)
        print('Decision Tree training code for Intro2ML coursework 1')
        print('-'*53+'\n')        
        path=input('Input path to dataset\n')
        data = np.loadtxt(path)
        print('Please select model\n')
        model=input('0 for DT, 1 for DT + pruning, 2 CV + DT, 3 CV + DT + pruning, 4 depth question\n')
        #cases for each model
        if model=='0':
            while True:
                split=float(input('Enter training data split value, eg 0.7\n'))
                if split<0 or split>1:
                    print('Invalid split entered')
                else:
                    break
            limit=input('Please enter a depth limit for the decision tree, if you do not want a limit just press enter\n')
            if limit=='':
                print('Unlimited Depth')
                limit=None
            else:
                limit=int(limit)
            
            np.random.shuffle(data)
            train=data[:int(len(data)*split)]
            test=data[int(len(data)*split):]
             
            model=binarySearchTree(train,limit=limit)
            print('Max depth is',model.get_max_depth())        
            y_pred=model.predict(test[:,:-1])
            cm=ev.confusion_matrix(test[:,-1],y_pred,plot=True)
            
            
            ev.get_metrics(test[:,-1], y_pred, printout=True)            
            r=ev.avg_recall_precision(cm)        
            print('\n')
            input('To restart, hit enter\n')
        if model=='1':
            split=float(input('Enter training data split value, eg 0.7\n'))
            while True:
                if split<0 or split>1:
                    print('Invalid split entered')
                else:
                    break
            print('You have entered '+str(split)+'\n')
            print('A validation set of '+str(round((1-split)/2,2))+' will be used\n')
            limit=input('Please enter a depth limit for the decision tree, if you do not want a limit just press enter\n')
            if limit=='':
                print('Unlimited Depth')
                limit=None
            else:
                limit=int(limit)            
            np.random.shuffle(data)
            t_split=split
            v_split=(1+split)/2
            train=data[:int(len(data)*t_split)]
            validation=data[int(len(data)*t_split):(int(len(data)*v_split))]
            test=data[int(len(data)*v_split):]

            model=binarySearchTree(train,limit=limit)
            print('Max depth before pruning:',model.get_max_depth())
            y_pred=model.predict(test[:,:-1])
            #evaluate
            cm=ev.confusion_matrix(test[:,-1],y_pred,plot=True,title='Unpruned')
            ev.get_metrics(test[:,-1], y_pred, printout=True)            
            r=ev.avg_recall_precision(cm)       
            
            print('\nPruning...\n')
            
            model.prune_tree(validation)
            print('Max depth after pruning:',model.get_max_depth())            
            y_pred=model.predict(test[:,:-1])
            #evaluate
            cm=ev.confusion_matrix(test[:,-1],y_pred,plot=True,title='Pruned')

            ev.get_metrics(test[:,-1], y_pred, printout=True)
            print('\n')
            input('To restart, hit enter\n')
            
        if model=='2':
            pass
        if model=='3':
            pass
        
        if model=='4':
            print('Training...\n')
            dp.plot_both(data)
            print('Training complete\n')
            input('To restart, hit enter\n')            
            
        
