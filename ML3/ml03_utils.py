import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE = 'data_ai_course_2015_2018.csv'


def load_data_for_univariate_regression():
    """Load csv and return it as numpy arrays."""

    df = pd.read_csv(DATA_FILE)

    X = df.hours_exam.values[:, np.newaxis]
    y = df.points_exam.values
    return X, y


def load_data_for_multivariate_regression():
    """Load csv and return it as numpy arrays."""

    df = pd.read_csv(DATA_FILE)
    X = df.xs(['hours_exam', 'hours_per_week', 'hours_per_week_ai', 'lecture_attendance'], axis=1).values
    y = df.points_exam.values
    return X, y


def load_data_2d_classification():
    """ Load data for binary classification """

    df = pd.read_csv(DATA_FILE)

    X = df.xs(['hours_exam', 'hours_per_week_ai'], axis=1).values

    df['y'] = 0 * df['points_exam'].values
    df.loc[df['points_exam'] < 30, 'y'] = 1

    y = df.y.values

    return X, y


def plot_xy_data(x, y, xlbl='', ylbl=''):
    """Create plot of x,y pairs."""
    plt.figure()
    plt.scatter(x, y)
    plt.xlabel(xlbl, fontsize=16)
    plt.ylabel(ylbl, fontsize=16)


def plot_1d_regr_model(model, c='b'):
    """Display predictions of the model in the x-range of an existing figure."""
    # Get the limits of axes in the current graph
    ax = plt.axis()
    # Build a set of points for which we'd like to display the predictions
    Xtst = np.linspace(ax[0], ax[1], 101)
    Xtst = Xtst[:, np.newaxis]
    # Compute the predictions
    ytst = model.predict(Xtst)
    # <YOUR CODE HERE>
    # Plot them
    plt.plot(Xtst, ytst, lw=3, c=c)
    plt.axis(ax)


def plot_xy_classified_data(x, y, c, xlbl='', ylbl='', colors=None):
    """Create plot of x,y pairs with classes c indicated by different colors."""
    if colors is None:
        colors = ['b', 'r', 'g', 'y', 'c', 'm']
    unique_classes = set(c)
    for k in unique_classes:
        plt.scatter(x[c == k], y[c == k], c=colors[k], s=36)
    plt.xlabel(xlbl, fontsize=16)
    plt.ylabel(ylbl, fontsize=16)


def plot_2d_class_model(model):
    """Plot the predictions and decision boundary of the model.

    Assumes that a plot of data points already exists.
    """
    ax = plt.axis()
    x1 = np.linspace(ax[0], ax[1], num=101)
    x2 = np.linspace(ax[2], ax[3], num=101)
    mx1, mx2 = np.meshgrid(x1, x2)
    sh = mx1.shape
    vx1 = np.reshape(mx1, (-1, 1))
    vx2 = np.reshape(mx2, (-1, 1))
    vx = np.hstack((vx1, vx2))
    vyhat = model.predict(vx)
    myhat = np.reshape(vyhat, sh)
    plt.contourf(mx1, mx2, myhat, cmap=plt.cm.cool, alpha=0.3)


def compute_err_mse(y, yhat):
    """Compute the mean squared error from the predictions and true values."""
    diffs = y - yhat
    total_err = diffs.dot(diffs.T)
    return total_err / y.shape[0]


def compute_err_01(y, yhat):
    """Compute the zero-one error, i.e. the mis-classification rate."""
    #raise NotImplementedError
    # <YOUR CODE HERE>
    return sum(abs(y - yhat))/y.shape[0]


def compute_model_error(model, X, y, err_func):
    """Compute the error of the model using the given data and error function."""
    #raise NotImplementedError
    # <YOUR CODE HERE>
    yNew = model.predict(X)
    return err_func(y, yNew)
