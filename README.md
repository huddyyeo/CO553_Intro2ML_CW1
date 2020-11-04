# Introduction to Machine Learning

##Coursework 1

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

In each model, evaluation metrics such as accuracy, precision, recall and F1, will be returned along with several plots. Model 1 and 2 will return a confusion matrix as well as a visualisation of the tree (up till depth 4), before and after pruning. Model 5 will return 2 plots that show how performance vaaries with depth.

## Using the files by themselves

### Running the decision tree

Import trees.py. To train it on the data, run the following

```
model=trees.binarySearchTee(data)
```

where data is the dataset you would like to analyse.
There are several other functions 

