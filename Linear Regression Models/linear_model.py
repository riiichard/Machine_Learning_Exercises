import numpy as np
from numpy.linalg import solve
from findMin import findMin
from scipy.optimize import approx_fprime
import utils

# Ordinary Least Squares
class LeastSquares:
    def fit(self,X,y):
        self.w = solve(X.T@X, X.T@y)

    def predict(self, X):
        return X@self.w

# Least squares where each sample point X has a weight associated with it.
class WeightedLeastSquares(LeastSquares): # inherits the predict() function from LeastSquares
    def fit(self,X,y,z):
        ''' YOUR CODE HERE '''
        V = np.diag(z)
        self.w = solve(X.T@(V@X), X.T@(V@y))
       

class LinearModelGradient(LeastSquares):

    def fit(self,X,y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros((d, 1))

        # check the gradient
        estimated_gradient = approx_fprime(self.w, lambda w: self.funObj(w,X,y)[0], epsilon=1e-6)
        implemented_gradient = self.funObj(self.w,X,y)[1]
        if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
            print('User and numerical derivatives differ: %s vs. %s' % (estimated_gradient, implemented_gradient));
        else:
            print('User and numerical derivatives agree.')

        self.w, f = findMin(self.funObj, self.w, 100, X, y)

    def funObj(self,w,X,y):

        ''' MODIFY THIS CODE '''
        # Calculate the function value
        f = np.sum(np.log(np.exp(X@w - y) + np.exp(y - X@w)))

        # Calculate the gradient value
        # Method 1:
        n,d=X.shape
        Array = np.zeros(n)
        for i in range(n):
            Array[i]=(np.exp(w.T@X[i]-y[i])-np.exp(y[i]-w.T@X[i]))/(np.exp(w.T@X[i]-y[i])+np.exp(y[i]-w.T@X[i]))
        g = X.T@Array
        # Method 2:
        # g = X.T @ ((np.exp(X @ w - y) - np.exp(y - X @ w)) / (np.exp(X @ w - y) + np.exp(y - X @ w)))

        return (f,g)


# Least Squares with a bias added
class LeastSquaresBias:

    def fit(self,X,y):
        ''' YOUR CODE HERE '''
        n,d=X.shape
        ones = np.ones((n,1))
        Z = np.concatenate((ones, X),axis=1)
        # Z = np.append(ones, X, axis=1)
        self.w = solve(Z.T@Z, Z.T@y)
        

    def predict(self, X):
        ''' YOUR CODE HERE '''
        n,d=X.shape
        ones = np.ones((n,1))
        Z = np.concatenate((ones, X),axis=1)
        # Z = np.append(ones, X, axis=1)
        return Z@self.w

# Least Squares with polynomial basis
class LeastSquaresPoly:
    def __init__(self, p):
        self.leastSquares = LeastSquares()
        self.p = p

    def fit(self,X,y):
        ''' YOUR CODE HERE '''
        Z = self.__polyBasis(X)
        self.w = solve(Z.T@Z, Z.T@y)
        
    def predict(self, X):
        ''' YOUR CODE HERE '''
        Z = self.__polyBasis(X)
        return Z@self.w
        
        
    # A private helper function to transform any matrix X into
    # the polynomial basis defined by this class at initialization
    # Returns the matrix Z that is the polynomial basis of X.
    def __polyBasis(self, X):
        ''' YOUR CODE HERE '''
        n,d=X.shape
        p = self.p
        Z = np.ones((n,p+1))
        for power in range(p+1):
            Z[:,power] = X[:,0]**power
        return Z
            
        
        
        
