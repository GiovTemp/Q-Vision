# Q-Vision/qvision/training.py

import numpy as np
from .neuron import optimizer, spatial_loss_derivative
from .utils import loss, accuracy

def train(optimizer_name, weights, bias, trainImgs, trainLabels, testImgs, testLabels, num_epochs, lr_weights, lr_bias,
          num_shots, momentum, batch_size, ideal_conditions, non_ideal_parameters):
    """Train the model using the optimization function from neuron.py
    :param ideal_conditions:
    :param lr_bias:
    :param lr_weights:
    :param num_epochs:
    :param testLabels:
    :param testImgs:
    :param trainLabels:
    :param trainImgs:
    :param bias:
    :param weights:
    :param optimizer_name:
    :param num_shots:
    :param batch_size:
    :param momentum:
    :param size:
    """
    weights, bias, loss_history, test_loss_history, accuracy_history, test_accuracy_history = optimizer(optimizer_name,
        spatial_loss_derivative, weights, bias, trainLabels, testLabels, trainImgs, testImgs,
        num_epochs, lr_weights, lr_bias, num_shots, momentum, batch_size, ideal_conditions, non_ideal_parameters
    )
    return weights, bias, loss_history, test_loss_history, accuracy_history, test_accuracy_history
