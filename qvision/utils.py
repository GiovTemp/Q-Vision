# Q-Vision/qvision/utils.py

import numpy as np

def sig(x):
    """ Compute the sigmoid activation function, with input x. """
    y = -11*x + 5.5
    return 1/(1 + np.exp(y))

def sigPrime(x):
    """ Compute the sigmoid derivative, with input x. """
    return sig(x)*(1-sig(x))*11

def loss(output, target):
    """ Compute the binary cross-entropy between output and target. """
    return -target*np.log(output) - (1-target)*np.log(1-output)

def accuracy(outputs, targets):
    """ Compute the total accuracy of the thresholded outputs against targets. """
    threshold = 0.5
    predicted = np.reshape((outputs >= threshold).astype(int), (-1))
    true_positive = np.sum(targets == predicted)
    return true_positive / len(targets)