"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np
from scipy import stats
import utils

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X # just memorize the trianing data
        self.y = y 

    def predict(self, Xtest):
        R, C = Xtest.shape
        
        # Construct the squared distance matrix between every row of X and Xtest
        square_distance = utils.euclidean_dist_squared(Xtest, self.X)
        # Find the array of distance from an example in increasing order
        inc_distance = np.argsort(square_distance)
        
        #initialize prediction output
        y_pred = np.zeros(R)
        # Make prediction on every single test example
        for i in range(R):
            # Get the nearest k points for each example
            nearest_points = inc_distance[i,0:self.k]
            # count the number of label 0 and label 1
            counts = np.bincount(self.y[nearest_points],minlength=2)
            if counts[0] < counts [1]:
                y_pred[i] = 1
            
        return y_pred
            
        