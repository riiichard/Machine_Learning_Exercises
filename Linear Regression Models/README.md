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
