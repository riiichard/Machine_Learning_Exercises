# Random Forest
In this directory, a **Random Forest** ensemble method is implemented via **_Bootstrapping_** and **_Random Tree_**.

## random_forest
In this file, the RandomForest class takes in hyperparameters num_trees and max_depth, and fits num_trees random trees each with maximum depth max_depth. For prediction,
have all trees predict and then take the mode.

## random_stump
**Acknowledgement:** This file is copied from UBC CPSC 340 course material.

This file contains one class: RandomStumpInfoGain.

The **RandomStumpInfoGain** class only considers **√d** randomly-chosen features, using Information Gain as criteria. 

## random_tree
**Acknowledgement:** This file is copied from UBC CPSC 340 course material.

This file contains one class: RandomTree.

**RandomTree** class is the same as **DecisionTree** class except that it uses **_RandomStump_** instead of DecisionStump and it takes a **_bootstrap
sample_** of the data before fitting.
