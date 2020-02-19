""" mapping.py
BE3M33UI - Artificial Intelligence course, FEE CTU in Prague

Module containing mapping functions.
"""

import numpy as np
from itertools import combinations_with_replacement

# noinspection PyPep8Naming
class PolynomialMapping:

    def __init__(self, max_deg=2):
        self.max_deg = max_deg

    def fit(self, X, y=None):
        #raise NotImplementedError
        # <YOUR CODE HERE>
        return self

    def transform(self, X, y=None):
        """Map the input data x into space of polynomials.
        """
        #raise NotImplementedError
        # <YOUR CODE HERE>
        tmp = np.array(X)
        X2 = X
        for i in range (1, self.max_deg):
            X2 = X2 * X
            tmp = np.hstack((tmp, X2))
        return tmp


# noinspection PyPep8Naming
class FullPolynomialMapping:

    def __init__(self, max_deg=2):
        self.max_deg = max_deg

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, max_deg=2):
        """Map X to fully polynomial space, including all cross-products."""
        #raise NotImplementedError
        # <YOUR CODE HERE>
        perms = []
        for i in range(0, X.shape[1]):
            perms.append(i)

        combs = list(combinations_with_replacement(perms, max_deg))
        Xn = np.ones((len(X), len(combs)))
        for k in range(0, len(combs)):
            #Xn[:, k] = X[:, k]
            for i in range(0, len(combs[k])):
                Xn[:, k] = Xn[:,k] * X[:, combs[k][i]]
        #print(Xn)
        return (Xn)

