"""
ML02-2.py

B(E)3M33UI - Artificial Intelligence course, FEE CTU in Prague
Model evaluation and diagnostics

Petr Posik, Jiri Spilka, CVUT, Praha 2018
"""

import numpy as np
from matplotlib import pyplot as plt

from sklearn.utils import shuffle
from sklearn import model_selection
from sklearn import svm
from sklearn.datasets import make_hastie_10_2

import plotting


def generate_data(n1=500, n2=500, sigma=0.04):

    cov = [[sigma, 0], [0, sigma]]

    xn1 = np.random.multivariate_normal([+0.4, 0.7], cov, int(n1/2))
    xn2 = np.random.multivariate_normal([-0.3, 0.7], cov, int(n1/2))
    xn = np.vstack((xn1, xn2))

    xp1 = np.random.multivariate_normal([-0.7, 0.3], cov, int(n2/2))
    xp2 = np.random.multivariate_normal([+0.3, 0.3], cov, int(n2/2))
    xp = np.vstack((xp1, xp2))

    XX = np.vstack((xn, xp))
    yy = np.hstack((np.zeros((n1,), dtype=np.int8), np.ones((n2,), dtype=np.int8))).T

    return XX, yy


X, y = generate_data(n1=400, n2=400)

# alternatively you can play with hastie et al. dataset
# X, y = make_hastie_10_2(n_samples=1000)
# y = y.astype(int)
# X = X[:, 0:2]  # use only two dimensions (to plot decision boundary)

X, y = shuffle(X, y)

plotting.plot_xy_classified_data(X[:, 0], X[:, 1], y)

Xtr, Xtst, ytr, ytst = model_selection.train_test_split(X, y, test_size=0.20)

# Create a classifier
model = svm.SVC(kernel='rbf', C = 1.0, gamma=.99)

# FIXME Tasks 14-16 - Learning curves
# Construct the array of training set sizes we are interested in.
tr_sizes = np.arange(40, ytr.shape[0], 10)
# raise NotImplementedError
# <YOUR CODE HERE>

train_errors, test_errors = plotting.compute_learning_curve(model, tr_sizes, Xtr, ytr, Xtst, ytst)
plotting.plot_learning_curve(tr_sizes, train_errors, test_errors)

plt.figure()
model.fit(X, y)
plotting.plot_xy_classified_data(X[:, 0], X[:, 1], y)
plotting.plot_2d_class_model(model)
plt.title('Model on the complete dataset: ')

# FIXME Tasks 17-18 - Validation curve
#raise NotImplementedError
# <YOUR CODE HERE>
param_name = 'gamma'
param_range = np.array([0.001, 0.01, 0.1, 0.5, 0.8, 1.00])
train_errors, test_errors = plotting.compute_validation_curve(model, param_name, param_range, Xtr, ytr, Xtst, ytst)
plotting.plot_validation_curve(param_range, train_errors, test_errors)
plt.show()
#raise NotImplementedError
