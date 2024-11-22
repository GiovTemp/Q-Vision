# Q-Vision/qvision/utils.py

from tabulate import tabulate
import numpy as np

def sig(x, alfa=11, beta=5.5):
    """ Compute the sigmoid activation function, with input x. """
    y = -alfa * x + beta
    return 1/(1 + np.exp(y))

def sigPrime(x, alfa=11, beta=5.5):
    """ Compute the sigmoid derivative, with input x. """
    return sig(x, alfa, beta)*(1-sig(x, alfa, beta)) * alfa

def loss(output, target):
    """ Compute the binary cross-entropy between output and target. """
    return -target*np.log(output) - (1-target)*np.log(1-output)

def accuracy(outputs, targets):
    """ Compute the total accuracy of the thresholded outputs against targets. """
    threshold = 0.5
    predicted = np.reshape((outputs >= threshold).astype(int), (-1))
    true_positive = np.sum(targets == predicted)
    return true_positive / len(targets)

def print_parameters(parameters):
    print(tabulate(parameters, headers="firstrow", tablefmt="grid"))

