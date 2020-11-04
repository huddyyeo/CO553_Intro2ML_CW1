# Introduction to Machine Learning Coursework 1

by Hudson Yeo, Ling Yu Choi, Monika Jotautaite and Grzegorz Sarapata

## Running the code

Download all py files and place them in the same folder. 

### Main 

For a quick overview, please run decisionTree.py
This file contains all the other codes and will help you evaluate your dataset with the model of your choice.
When running the file, you will be asked to input a path to your dataset.
Then you will be asked to select the model you want.
Please input: 

1) 1, if you would like to run a decision tree only.
2) 2, if you would like to run a decision tree with pruning
3) 3, if you would like to run cross validation
4) 4, if you would like to run cross validation and pruning
5) 5, if you would like to run the code for the depth evaluation (depth question)

In case 1 and 2, you will be asked to input a value for the train/test split (from 0 to 1) as well as depth limit if you want. If you do not want a depth limit for the tree, please do hit enter without typing anything. For model 2, the original test set will be equally divided into 2, one for pruning and one for testing. ie, if you enter a train/test split of 0.8, it will prune on 10% and test on the other 10% of the dataset.

In each model, evaluation metrics such as accuracy, precision, recall and F1, will be returned along with several plots. Model 1 and 2 will return a confusion matrix as well as a visualisation of the tree (up till depth 4), before and after pruning. Model 5 will return 2 plots that show how performance varies with depth.

## Using the files by themselves

### Running the decision tree

Import trees.py. To train it on the data, run the following

```
model=trees.binarySearchTree(data)
```

where data is the dataset you would like to analyse.
There are several other functions:

```
model.get_max_depth()  #returns the depth of the tree
model.predict(test_set) #returns the output labels for a given test set
model.prune_tree(validation_set) #returns the output labels for a given validation set
model.visualise_tree() #plots a visualisation of the tree up till depth 4
```

### Evaluating results

Import evaluation.py as ev. To evaluate results from the decision tree, please ensure you first have a test set and a set of predicted labels. 

```
cm=ev.confusion_matrix(test[:,-1],y_pred) #gets conf matrix
i=ev.get_metrics(cm,printout=True) #gets precision, recall, F1 and accuracy
ev.plot_conf_matrix(cm) #plot conf matrix
```

The above will work for most purposes but if you would like individual scores, please use the following
```
precision,recalls=ev.get_recalls_precision(cm)
f1=ev.get_f1_scores(cm)
accuracy=ev.get_class_rate(cm)
```

### Cross Validating

Import cross_validation.py as cv. 

For 10 fold cross validation, run the following
```
results=cv.grow_binary_trees(data)
```

For 10 fold nested cross validation with pruning, run the following
```
results=cv.grow_binary_trees(data,pruning=True)
```

To evaluate results, please use the following
```
results=ev.get_averages(results) #returns a dict of results
#or
results=ev.print_results(results) #results a dict of results with appropriate CIs
```

Lastly, to obtain a plot to compare pruning results on both clean and noisy dataset, you may also use 
```
cv.metrics_pruning_plot(results_clean, results_clean_pruned, results_noisy, results_noisy_pruned)
```

### Depth analysis

Import depth_question.py as dp and use
```
dp.plot_both(data) #plots the diagrams seen in the report
```
There are several other functions which are
```
dp.get_values(data) #runs tree 20 times, with increasing depth limit and returns depths, accuracies, pruned_depths and pruned_accuracies
dp.graph_depths(data) #runs tree 20 times and plots depth changes before and after pruning
dp.graph_depth_accuracy(data) #runs tree 20 times and plots accuracy against tree depth
```

