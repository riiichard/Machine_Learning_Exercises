import numpy as np
import utils
import random
from decision_stump import DecisionStumpInfoGain
from random_tree import RandomTree


class RandomForest():
        
    def __init__(self, num_trees, max_depth):
        self.num_trees = num_trees
        self.max_depth = max_depth
    
    
    def fit(self, X, y):
        # Train data on each tree
        # initialize a forest
        self.random_forest = []
        for tree in range(self.num_trees):
            # Bootstrapping and Random Trees Step
            one_tree = RandomTree(max_depth = self.max_depth)
            one_tree.fit(X,y)
            self.random_forest.append(one_tree)
        
    
    def predict(self, X):
        D, N = X.shape
        # Predict on each tree
        # initialize forest of predictions 
        forest_pred = np.ones((D, self.num_trees))
        for one_tree in range(self.num_trees):
            # prediction on one tree for all examples
            # every column is a prediction for a tree
            forest_pred[:,one_tree] = self.random_forest[one_tree].predict(X)
        # Average the prediction forest for each example
        avg_forest_pred = np.ones(D)
        for ex in range(D):
            avg_forest_pred[ex] = utils.mode(forest_pred[ex,:])
        
        return avg_forest_pred
        
    
    