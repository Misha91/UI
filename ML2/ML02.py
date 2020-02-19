# -*- coding: utf-8 -*-
"""
    ML02.py
    ~~~~~~~

    Linear regression
    B(E)3M3UI - Artificial Intelligence

    :author: Petr Posik, Jiri Spilka, 2019

    FEE CTU in Prague
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

import linreg


# # Load data for simple regression
X, y = linreg.load_data_for_univariate_regression()
print('\n==Simple regression==')
print('The shapes of X and y are:')
print(X.shape)
print(y.shape)

#print(X)
#print(linreg.homogenize(X))
# # Plot the input data
linreg.plot_2d_data(X, y)


# FIXME Task 1: Estimate the model parameters by hand. <PUT YOUR ESTIMATE OF w_0 AND w_1 HERE>
w_0 = 20
w_1 = 0.92
wguess = np.array([w_0, w_1])
linreg.plot_predictions(wguess, c='g', label='guess')

# # Compute the error of the model
print('\nThe hand-crafted coefficients:')
print(wguess)
err = linreg.compute_cost_regr_lin(wguess, X, y)
print('The error:', err)

# # Find optimal weights by minimization of J
w1 = linreg.fit_regr_lin_by_minimization(X, y)
print('\nCoefficients found by minimization of J:')
print(w1)
err = linreg.compute_cost_regr_lin(w1, X, y)
print('The error:', err)
linreg.plot_predictions(w1, c='b', label='minimize')

# # Find optimal weights by normal equation
w2 = linreg.fit_regr_lin_by_normal_equation(X, y)
print('\nCoefficients found by normal equation:')
print(w2)
err = linreg.compute_cost_regr_lin(w2, X, y)
print('The error:', err)
linreg.plot_predictions(w2, c='r', label='normal eq.')
plt.show()
# # Use scikit-learn package to train the model
# FIXME Task 8: Linear regression using scikit
# <ADD THE CODE WHICH CREATES AN INSTANCE OF LINEAR REGRESSION,
# AND TRAINS IT ON THE TRAINING DATA>

lr = linear_model.LinearRegression().fit(X, y)
w3 = np.array([lr.intercept_, lr.coef_[0]])
print('\nCoefficients found by sklearn.linear_model.LinearRegression:')
print(w3)
err = linreg.compute_cost_regr_lin(w3, X, y)
print('The error:', err)

# # Multivariate regression
X, _ = linreg.load_data_for_multivariate_regression()
print('\n==Multivariate regression==')
print('The shapes of X and y are:')
print(X.shape)
print(y.shape)

# FIXME Task 9: Multivariate regression
# # Fit the regression model by any method

X, y = linreg.load_data_for_multivariate_regression()
lr = linear_model.LinearRegression().fit(X, y)
wmulti = np.array([lr.intercept_, lr.coef_[0]])
print('\nCoefficients for multivariate regression model:')
print(wmulti)
err = np.mean((y - lr.predict(X).T)**2)
print('The error:')
print(err)

plt.show()