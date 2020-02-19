"""
B(E)3M33UI - Artificial Intelligence course, FEE CTU in Prague
Neural Networks

Petr Posik, CVUT, Praha 2017
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
import random
from plotting import plot_2D_class_model, plot_xy_classified_data

def sigm(x):
    return 1 / (1 + np.exp(-x))

def main():

    # Load data
    data = np.loadtxt('data_1.csv', delimiter=',')
    # data = np.loadtxt('data_xor_rect.csv', delimiter=',')

    print(data.shape)

    X = data[:, 0:2]
    y = data[:, 2].astype(int)
    plot_xy_classified_data(X[:, 0], X[:, 1], y)
    print('\nThe dataset (X and y) size:')
    print('X: ', X.shape)
    print('y: ', y.shape)
    print('sum(y = 0): ', sum(y == 0))
    print('sum(y = 1): ', sum(y == 1))

    epochs = 5000
    alpha = 20
    tol = 0.0001

    w11 = random.uniform(0, 1)/100
    w12 = random.uniform(0, 1)/100
    w13 = random.uniform(0, 1)/100
    w21 = random.uniform(0, 1)/100
    w22 = random.uniform(0, 1)/100
    w23 = random.uniform(0, 1)/100
    w31 = random.uniform(0, 1)/100
    w32 = random.uniform(0, 1)/100
    w33 = random.uniform(0, 1)/100
    print(w11,w12,w13,w21,w22,w23,w31,w32,w33)
    for e in range(epochs):
        for i in range(0, X.shape[0]):
            x1 = X[i,0]
            x2 = X[i,1]
            a1 = x1 * w11 + x2 * w12 + w13
            a2 = x1 * w21 + x2 * w22 + w23
            z1 = sigm(a1)
            z2 = sigm(a2)
            a3 = z1 * w31 + z2 * w32 + w33
            z3 = sigm(a3)
            b3 = z3 * (1 - z3) * (y[i] - z3)
            dw31 = z1*b3
            dw32 = z2*b3
            dw33 = b3
            dw13 = z1 * (1 - z1) * w31 * b3
            dw11 = x1 * dw13
            dw12 = x2 * dw13
            dw23 = z2 * (1 - z2) * w32 * b3
            dw21 = x1 * dw23
            dw22 = x2 * dw23
            #print(z3, y[i])
            #print(x1, x2, a1, a2, z1, z2, z3)
            #print("W",w11, w12, w13, w21, w22, w23, w31, w32, w33)
            #print(dw11, dw12,dw13,dw21,dw22,dw23,dw31,dw32,dw33)

            w11 += alpha * dw11
            w12 += alpha * dw12
            w13 += alpha * dw13
            w21 += alpha * dw21
            w22 += alpha * dw22
            w23 += alpha * dw23
            w31 += alpha * dw31
            w32 += alpha * dw32
            w33 += alpha * dw33

        diff = 0
        for i in range(0, X.shape[0]):
            x1 = X[i,0]
            x2 = X[i,1]
            a1 = x1*w11 + x2*w12 + w13
            a2 = x1*w21 + x2*w22 + w23
            z1 = sigm(a1)
            z2 = sigm(a2)
            a3 = z1*w31 + z2*w32 + w33
            z3 = sigm(a3)
            diff += (y[i] - z3)**2
            #print(y[i], z3)
        diff /= X.shape[0]
        print(diff)
        if (diff <= tol): break

    for i in range(0, X.shape[0]):
        x1 = X[i,0]
        x2 = X[i,1]
        a1 = x1*w11 + x2*w12 + w13
        a2 = x1*w21 + x2*w22 + w23
        z1 = sigm(a1)
        z2 = sigm(a2)
        a3 = z1*w31 + z2*w32 + w33
        z3 = sigm(a3)
        diff += (y[i] - z3)**2


    # # FIXME Task 6: Implement backpropagation
    # raise NotImplementedError
    # <YOUR CODE HERE>

    # # FIXME Task 7: Visualize decision boundary
    # raise NotImplementedError
    clf = MLPClassifier((20,20,10))
    clf.fit(X, y)
    print(clf.score(X, y))
    plot_2D_class_model(clf)
    # <YOUR CODE HERE>
    plt.show()
    # # FIXME Task 8: MLPClassifier
    # raise NotImplementedError
    # <YOUR CODE HERE>

    # # FIXME Task 9: MLPClassifier
    # raise NotImplementedError
    # <YOUR CODE HERE>
# catch all exceptions
try:
    plt.ion()  # turn interactive mode on so our plots stay open like in matlab
    main()

except BaseException as e:
    import traceback
    traceback.print_exc()  # mimic printing traceback from an exception
    plt.ioff()  # turn interactive mode off
    plt.show()  # show whatever we have to show, it will stay open because we turned interactive mode off
