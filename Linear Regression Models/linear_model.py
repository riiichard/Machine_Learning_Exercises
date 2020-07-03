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
            
# Logistic Regression
class logReg:
    def __init__(self, verbose=1, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self,X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)
    def predict(self, X):
        return np.sign(X@self.w)

# Logistic Regression with L2-Regularization
class logRegL2:
    def __init__(self, lammy=1.0, verbose=1, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True
        self.lammy = lammy

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)
        w2 = w.T.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))
        # Update to L2-Regularization
        f = f + 0.5 * self.lammy * w2

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res) + self.lammy * w

        return f, g

    def fit(self,X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)
    def predict(self, X):
        return np.sign(X@self.w)

# Logistic Regression with L1-Regularization
class logRegL1:
    def __init__(self, L1_lambda=1.0, verbose=1, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True
        self.L1_lambda = L1_lambda

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self,X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMinL1(self.funObj, self.w, self.L1_lambda, 
                                      self.maxEvals, X, y, verbose=self.verbose)
    def predict(self, X):
        return np.sign(X@self.w)

# Logistic Regression with L0-Regularization
class logRegL0(logReg):
    def __init__(self, L0_lambda=1.0, verbose=2, maxEvals=400):
        self.verbose = verbose
        self.L0_lambda = L0_lambda
        self.maxEvals = maxEvals

    def fit(self, X, y):
        n, d = X.shape
        minimize = lambda ind: findMin.findMin(self.funObj,
                                                  np.zeros(len(ind)),
                                                  self.maxEvals,
                                                  X[:, ind], y, verbose=0)
        selected = set()
        selected.add(0)
        minLoss = np.inf
        oldLoss = 0
        bestFeature = -1

        while minLoss != oldLoss:
            oldLoss = minLoss
            print("Epoch %d " % len(selected))
            print("Selected feature: %d" % (bestFeature))
            print("Min Loss: %.3f\n" % minLoss)

            for i in range(d):
                if i in selected:
                    continue

                selected_new = selected | {i} # tentatively add feature "i" to the seected set

                # TODO for Q2.3: Fit the model with 'i' added to the features,
                # then compute the loss and update the minLoss/bestFeature
                # Fit the model with updated features
                w, f = minimize(list(selected_new))
                # Compute the loss
                newLoss = f + self.L0_lambda * len(w[w!=0])
                # Update the minLoss/bestFeature
                if newLoss < minLoss:
                    minLoss = newLoss
                    bestFeature = i

            selected.add(bestFeature)

        self.w = np.zeros(d)
        self.w[list(selected)], _ = minimize(list(selected))        
        
# One-vs-all Logistic Regression
class logLinearClassifier:
    def __init__(self, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True

    # REFERENCE: copy from logReg above
    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g
    
    # Fit via One-vs-all and Logistic Regression
    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes,d))
        
        # calculate wtx for each class
        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1

            # solve for the minimizer parameter by gradient descent
            self.W[i], f = findMin.findMin(self.funObj, self.W[i],
                                      self.maxEvals, X, ytmp, verbose=self.verbose)

    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)
    
# Logistic Regression with softmax loss
class softmaxClassifier:
    def __init__(self, verbose=0, maxEvals=100):
            self.verbose = verbose
            self.maxEvals = maxEvals

    def funObj(self, w, X, y):
        n, d = X.shape
        
        # Prepare the values
        # reshape the w from vector to matrix
        w_matrix = np.reshape(w, (self.n_classes, d))
        # sum across all the classes
        # wx = X @ (w_matrix.T)
        wx = np.dot(X, w_matrix.T)
        sum_class = np.sum(np.exp(wx), axis=1)
        # select wx with corresponding y class
        wy = np.zeros((n,))
        indicator = np.zeros((n,self.n_classes))
        for row in range(n):
            wy[row] = wx[row,y[row]]
            indicator[row,y[row]] = 1
            
        # Calculate the function value
        f = np.sum(np.log(sum_class) - wy)

        # Calculate the gradient value
        grad = ((np.exp(wx) / sum_class[:,None] - indicator).T) @ X
        g = grad.flatten()

        return f, g

    def fit(self, X, y): 
        n, d = X.shape
        
        # Initialize parameter matrix W
        y_labels = np.unique(y)
        k = len(y_labels)
        self.n_classes = len(y_labels)
        self.w = np.zeros(k*d) 
        # Check gradient
        utils.check_gradient(self, X, y)
        # Fit the model
        w_vector, f = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)
        
        self.W = np.reshape(w_vector, (k,d))
    
    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)
    
class KernelBasis:
    def kernel_RBF(X1, X2, sigma=1):
        n1, d1 = X1.shape
        n2, d2 = X2.shape
        # ||xi - xj|| = xiTxi - 2xiTxj + xjTxj, (n1 x n2)
        # construct xiTxi, columns are identical
        xiTxi = X1**2 @ np.ones((d1,n2))
        # construct xiTxj
        xiTxj = X1 @ X2.T
        # construct xjTxj, rows are identical
        xjTxj = np.ones((n1,d2))@(X2.T**2)
        # compute ||xi - xj||, Shape n1 x n2
        norm = xiTxi - 2*xiTxj + xjTxj
        # compute kij
        k = np.exp(- norm / (2 * (sigma**2)))
        return k

    def kernel_poly(X1, X2, p=2):
        k = (1 + X1 @ (X2.T))**p
        return k
