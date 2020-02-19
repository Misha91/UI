# -*- coding: utf-8 -*-
"""
    ML03-1.py
    ~~~~~~~~~

    Basis expansion with linear regression
    B(E)3M3UI - Artificial Intelligence

    :author: Petr Posik, Jiri Spilka, 2019

    FEE CTU in Prague
"""

from matplotlib import pyplot as plt
from sklearn import linear_model, pipeline

import mapping
import ml03_utils


# # Load data for simple regression
X, y = ml03_utils.load_data_for_univariate_regression()
print('\n== Simple regression ==')
print('The shapes of X and y are:')
print(X.shape)
print(y.shape)

# # Plot the input data
ml03_utils.plot_xy_data(X, y, xlbl='hours_exam', ylbl='points_exam')

# # Fit linear regression model and plot it
lm = linear_model.LinearRegression()
lm.fit(X, y)
ml03_utils.plot_1d_regr_model(lm)

# # Compute its error
err = ml03_utils.compute_model_error(lm, X, y, ml03_utils.compute_err_mse)
print('Degree 1 model: MSE = {:.3f}'.format(err))

#print(X, pm.transform(X))


print(err)
plt.show()


# # Fit a quadratic model and plot it
#raise NotImplementedError
# <YOUR CODE HERE>

# # Higher degree polynomials
ml03_utils.plot_xy_data(X, y, xlbl='disp', ylbl='hp')

# Degrees of polynomials with the colors of the lines
degrees = [(1, 'b'), (2, 'r'), (3, 'g'), (4, 'y')]
# Array for legend descriptions
legstr = []
for deg, color in degrees:
    #raise NotImplementedError
    pm = mapping.PolynomialMapping(deg)
    pip = pipeline.Pipeline([('pm', pm), ('lm', lm)])
    pip.fit(X, y)
    ml03_utils.plot_1d_regr_model(pip, color)
    err = ml03_utils.compute_model_error(pip, X, y, ml03_utils.compute_err_mse)
    print("Degree ", deg, " error is ", err)
plt.legend(legstr, loc='upper left')
plt.show()