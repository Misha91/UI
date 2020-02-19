# -*- coding: utf-8 -*-
"""
    ML04-1.py
    ~~~~~~~~~

    Model evaluation and diagnostics
    B(E)3M3UI - Artificial Intelligence

    :author: Petr Posik, Jiri Spilka, 2019

    FEE CTU in Prague
"""

import numpy as np
import pandas as pd
from sklearn import linear_model, metrics, svm, preprocessing, model_selection
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import plotting


DATA_FILE = 'data_ai_course_2015_2018.csv'


def load_data_binary_classification():
    """ Load data for binary classification """

    df = pd.read_csv(DATA_FILE)
    X = df.xs(['hours_exam', 'hours_per_week', 'hours_per_week_ai', 'lecture_attendance',
               'points_st1', 'points_st2', 'seminar_attendance'], axis=1).values
    X = X.astype(np.float)

    df['y'] = 0 * df['points_exam'].values
    df.loc[df['points_exam'] < 40, 'y'] = 1

    y = df.y.values

    return X, y


# Load data
X, y = load_data_binary_classification()
print('\nThe dataset (X and y) size:')
print('X: ', X.shape)
print('y: ', y.shape)

# scale the data
sc = preprocessing.StandardScaler()
X = sc.fit_transform(X)

# FIXME Task 1: Implement metrics to measure model performance
clf = SVC(kernel='poly', gamma=1, probability=True)
y_hat = clf.fit(X, y).predict(X)
print(metrics.confusion_matrix(y, y_hat))
A = {}
A['clf'] = clf
A['descr'] = "SVC poly"
A['color'] = 'r'

# <YOUR CODE HERE>
# clf = ... classifier
# clf.fit, clf.predict

# FIXME Tasks 2-3: ROC curves
#fpr, tpr, thresholds = metrics.roc_curve(y, y_hat, pos_label=2)
# # Setup the list of models
models = []
models.append(A)

clf = SVC(kernel='linear', gamma=1, probability=True)
y_hat = clf.fit(X, y).predict(X)
print(metrics.confusion_matrix(y, y_hat))
A = {}
A['clf'] = clf
A['descr'] = "SVC Lin"
A['color'] = 'g'
models.append(A)
# Setup a classifier
# They can be even pipelines with basis expansion, etc.
clf = svm.SVC(kernel='rbf', gamma=1, probability=True)
models.append({'clf': clf, 'descr': 'SVM rbf', 'color': 'b'})

#raise NotImplementedError
# <YOUR CODE HERE>

# use the models, fit them and plot ROC
plotting.plot_roc(models, X, y, X, y)

# FIXME Tasks 4-5: Model evaluation train/test split

#raise NotImplementedError
# <YOUR CODE HERE>

Xtr, Xtst, ytr, ytst = train_test_split(X, y, test_size=0.5)

print('\nShapes of training and testing X:')
print(Xtr.shape)
print(Xtst.shape)
print('Shapes of training and testing y:')
print(ytr.shape)
print(ytst.shape)

# Fit the model to training data, validate on test data
plotting.plot_roc(models, Xtr, ytr, Xtst, ytst)
# <YOUR CODE HERE>
clf.fit(Xtr, ytr)
clf.predict(Xtst)
acc_tr = metrics.accuracy_score(ytr, clf.predict(Xtr))
acc_tst = metrics.accuracy_score(ytst, clf.predict(Xtst))
print('\nThe training and testing accuracies when simple split is used:')
print('train acc:', acc_tr)
print('test  acc:', acc_tst)

# FIXME Tasks 6-10: Cross-validation and manual tuning
# Use 5-fold cross-validation
# Play with SVM parameters and try to understand them
folds = 5
accuracies = cross_val_score(models[1]['clf'], X, y, cv=folds)
# <YOUR CODE HERE>

print("\n{:d}-cross-validation ACC: {:.2f} (+/- {:.2f})".format(folds, accuracies.mean(), accuracies.std()))
print('accuracies:', accuracies)

# FIXME Tasks 11-13: Cross-validation and grid search
#raise NotImplementedError
# <YOUR CODE HERE>
clf = svm.SVC(kernel='rbf', probability=True)
parameters = {
    'C': (1.0, 0.9, 0.8),
    'gamma': (1, 'auto')
}

clf = GridSearchCV(clf, parameters, cv=5, n_jobs=-1, verbose=1)

print('\nTUNING PARAMETERS:\n')
for param_name in sorted(parameters.keys()):
    print(param_name, parameters[param_name])
clf.fit(Xtr, ytr)
print("\nAFTER TUNING: ")
print('Best parameters found:')
print(clf.best_params_)
print('CV score for the best parameter values:')
print(clf.best_score_)

# Print the scores of the TUNED classifier for training and testing data
#raise NotImplementedError
# <YOUR CODE HERE>
clf.predict(Xtst)
acc_tr = metrics.accuracy_score(ytr, clf.predict(Xtr))
acc_tst = metrics.accuracy_score(ytst, clf.predict(Xtst))
print('\nThe training and testing accuracies for TUNED classifier:')
print('train acc:', acc_tr)
print('test  acc:', acc_tst)
