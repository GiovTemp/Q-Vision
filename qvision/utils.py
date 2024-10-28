from tabulate import tabulate
import cupy as cp  # Importing CuPy

def sig(x):
    """ Compute the sigmoid activation function, with input x. """
    y = -11 * x + 5.5
    return 1 / (1 + cp.exp(y))  # Using cp.exp

def sigPrime(x):
    """ Compute the sigmoid derivative, with input x. """
    return sig(x) * (1 - sig(x)) * 11

def loss(output, target):
    """ Compute the binary cross-entropy between output and target. """
    return -target * cp.log(output) - (1 - target) * cp.log(1 - output)  # Using cp.log

def accuracy(outputs, targets):
    """ Compute the total accuracy of the thresholded outputs against targets. """
    threshold = 0.5
    predicted = cp.reshape((outputs >= threshold).astype(int), (-1))  # Using cp.reshape
    true_positive = cp.sum(targets == predicted)  # Using cp.sum
    return true_positive / len(targets)

def print_parameters(parameters):
    print(tabulate(parameters, headers="firstrow", tablefmt="grid"))