""" ML03-2.py
BE3M33UI - Artificial Intelligence course, FEE CTU in Prague

Basis expansion for linear classification. SVM.
"""

from matplotlib import pyplot as plt
from sklearn import linear_model, pipeline, svm
from sklearn.linear_model import LogisticRegression
import mapping
import ml03_utils
from sklearn.svm import SVC
# # Load 2D data for classification
X, y = ml03_utils.load_data_2d_classification()
ml03_utils.plot_xy_classified_data(X[:, 0], X[:, 1], y, xlbl='hours_exam', ylbl='hours_per_week_ai')

print('\n=== Comparison of several classification models ===\n')

# # Logistic regression model with original features only
#raise NotImplementedError
# <YOUR CODE HERE>
logr = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X, y)
logr.predict_proba(X[:2, :])
ml03_utils.plot_2d_class_model(logr)
#plt.show()
# # The error of the model
err = ml03_utils.compute_model_error(logr, X, y, ml03_utils.compute_err_01)
# Build the message and display it
msg = 'Pure logistic regression: error = {:.3f}'.format(err)
plt.title(msg)
print(msg)

# # Logistic regression with purely polynomial terms
plt.figure()
ml03_utils.plot_xy_classified_data(X[:, 0], X[:, 1], y, xlbl='hours_exam', ylbl='hours_per_week_ai')
degree = 2

#raise NotImplementedError
# <YOUR CODE HERE>
pm = mapping.PolynomialMapping(degree)
#print(X, pm.transform(X))
pip = pipeline.Pipeline([('pm', pm), ('logr', logr)])
pip.fit(X, y)
ml03_utils.plot_2d_class_model(pip)
err = ml03_utils.compute_model_error(pip, X, y, ml03_utils.compute_err_mse)
# Build the message and display it
msg = ('Logistic regression with pure polynomials (deg = {:d}): '
       'error = {:.3f}').format(degree, err)


plt.title(msg)
print(msg)


# # Logistic regression with fully polynomial mapping
plt.figure()
ml03_utils.plot_xy_classified_data(X[:, 0], X[:, 1], y, xlbl='hours_exam', ylbl='hours_per_week_ai')
degree = 2
pm = mapping.FullPolynomialMapping(degree)
pip = pipeline.Pipeline([('pm', pm), ('logr', logr)])
pip.fit(X, y)
ml03_utils.plot_2d_class_model(pip)
#raise NotImplementedError
# <YOUR CODE HERE>
err = ml03_utils.compute_model_error(pip, X, y, ml03_utils.compute_err_mse)
# Build the message and display it
msg = ('Logistic regression with full polynomial mapping (deg = {:d}): '
       'error = {:.3f}').format(degree, err)
plt.title(msg)
print(msg)

# # Support vector classification with linear kernel
plt.figure()
ml03_utils.plot_xy_classified_data(X[:, 0], X[:, 1], y, xlbl='hours_exam', ylbl='hours_per_week_ai')

#raise NotImplementedError
# <YOUR CODE HERE>

clf = SVC(kernel='linear')
clf.fit(X, y)
ml03_utils.plot_2d_class_model(clf)
err = ml03_utils.compute_model_error(clf, X, y, ml03_utils.compute_err_mse)
# Build the message and display it
msg = 'SVM with linear kernel: error = {:.3f}'.format(err)
plt.title(msg)
print(msg)

# # Support vector classification with RBF kernel
plt.figure()
ml03_utils.plot_xy_classified_data(X[:, 0], X[:, 1], y, xlbl='hours_exam', ylbl='hours_per_week_ai')


#raise NotImplementedError
# <YOUR CODE HERE>
clf = SVC(gamma=1)
clf.fit(X, y)
ml03_utils.plot_2d_class_model(clf)
err = ml03_utils.compute_model_error(clf, X, y, ml03_utils.compute_err_mse)
# Build the message and display it
msg = 'SVM with RBF kernel: error = {:.3f}'.format(err)
plt.title(msg)
print(msg)

plt.show()
