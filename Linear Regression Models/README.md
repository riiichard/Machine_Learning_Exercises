# Linear Regression Models
The linear_model includes implementations of some classic linear models used in regression and classification.

## LeastSquares
**Acknowledgement:** This class is copied from UBC CPSC 340 course material.

## WeightedLeastSquares
Least squares where each example X has a weight associated with it. **predict** function is as before.

## LinearModelGradient
The class **LinearModelGradient** gives a method/algorithm of **Smooth Approximation to the L1-Norm**. It implements a gradient-based strategy for fitting the robust regression model under the log-sum-exp approximation.

## LeastSquaresBias
Based on the **LeastSquares** class, it adds a bias variable (also called an intercept).

## LeastSquaresPoly (transform to fit non-linear)
This class wants to fit non-linear data using a linear model by transforming the original basis into a polynomial basis.

## logReg
**Acknowledgement:** This class is copied from UBC CPSC 340 course material.

This class is an implementation of Logistic Regression

## logRegL2
It fits a **Logistic** Regression model with **L2-regularization**.

## logRegL1
It fits a **Logistic** Regression model with **L1-Regularization**.

## logRegL0
It fits a **Logistic** Regression model with **L0-regularization**. Regularization step is using **forward selection algorithm**.

## logLinearClassifier
This class replaces the squared loss in the one-vs-all model with the logistic loss, fitting **One-vs-all Logistic Regression**.

## softmaxClassifier
It fits W using the softmax loss instead of fitting k independent classifiers.

## kernel_RBF
It fits a model transforming the original basis data into **RBF kernel** basis for **Logistic Regression**.

## kernel_poly
It fits a model transforming the original basis data into **polynomial kernel** basis for **Logistic Regression**.

