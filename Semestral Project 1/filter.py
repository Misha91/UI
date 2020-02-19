"""
B(E)3M33UI - Support script for the first semestral task
"""

from sklearn.datasets import load_files
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

#in case stopwords are missing, uncomment 2 following lines:
#import nltk
#nltk.download('stopwords')


# you can do whatever you want with the these data
TR_DATA = 'data/spam-data-1'
TST_DATA = 'data/spam-data-2'


def modified_accuracy(y, y_pred):
    """Return a modified accuracy score with larger weight of false positives."""
    cm = confusion_matrix(y, y_pred)
    if cm.shape != (2, 2):
        raise ValueError('The ground truth values and the predictions may contain at most 2 values (classes).')
    return (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[1, 1] + 10 * cm[0, 1] + cm[1, 0])


our_scorer = make_scorer(modified_accuracy, greater_is_better=True)


def train_filter(X, y):
    """Return a trained spam filter.
    Please keep the same arguments: X, y (to be able to import this function for evaluation)
    """
    assert 'X' in locals().keys()
    assert 'y' in locals().keys()
    assert len(locals().keys()) == 2
    y2 = np.invert(y)

    parameters = {
        'vect__max_df': (0.75, 0.8, 0.85),
    }
    pipeline = Pipeline([
        ('vect', CountVectorizer(stop_words = stopwords.words('english'))),
        ('tfidf', TfidfTransformer(sublinear_tf=True)),
        ('clf', MLPClassifier((120, 120, 80, 80, 40))),])

    pipe = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1)
    pipe.fit(X, y)
    return pipe


def predict(filter1, X):
    """Produce predictions for X using given filter.
    Please keep the same arguments: X, y (to be able to import this function for evaluation)
    """
    assert len(locals().keys()) == 2

    return filter1.predict(X)

if __name__ == '__main__':

    # Demonstration how the filter will be used but you can do whatever you want with the these data
    # Load training data
    data_tr = load_files(TR_DATA, encoding='utf-8')
    X_train = data_tr.data
    y_train = data_tr.target

    # Load testing data
    data_tst = load_files(TST_DATA, encoding='utf-8')
    X_test = data_tst.data
    y_test = data_tst.target

    # or you can make a custom train/test split (or CV)
    X = X_train.copy()
    X.extend(X_test)
    y = np.hstack((y_train, y_test))

    # Train the filter

    #Taking 100% of data for training, Neural Network
    filter3 = train_filter(X, y)
    y_tr_pred = predict(filter3, X_train)
    print('Modified accuracy on training data: ', modified_accuracy(y_train, y_tr_pred))
    y_tst_pred = predict(filter3, X_test)
    print('Modified accuracy on testing data: ', modified_accuracy(y_test, y_tst_pred))
    """
    #Taking 70% of data for training, Neural Network
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_train, test_size=0.3)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X_test, y_test, test_size=0.3)
    filter2 = train_filter(X_train1 + X_train2, np.concatenate((y_train1, y_train2)))
    y_tr_pred = predict(filter2, X_train)
    print('Modified accuracy on training data: ', modified_accuracy(y_train, y_tr_pred))
    y_tst_pred = predict(filter2, X_test)
    print('Modified accuracy on testing data: ', modified_accuracy(y_test, y_tst_pred))
    """
    """
    #Taking 50% of data for training, Neural Network
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_train, test_size=0.5)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X_test, y_test, test_size=0.5)
    filter1 = train_filter(X_train2 + X_train1, np.concatenate((y_train2, y_train1)))
    y_tr_pred = predict(filter1, X_train)
    print('Modified accuracy on training data: ', modified_accuracy(y_train, y_tr_pred))
    y_tst_pred = predict(filter1, X_test)
    print('Modified accuracy on testing data: ', modified_accuracy(y_test, y_tst_pred))
    """
