# Q-Vision/qvision/training.py

import numpy as np
from .neuron import neuron, optimizer, spatial_loss_derivative
from .utils import loss, accuracy

def train(optimizer_name, weights, bias, trainImgs, trainLabels, testImgs, testLabels, num_epochs, lr_weights, lr_bias,
          num_shots, momentum, batch_size):
    """Train the model using the optimization function from neuron.py
    :param momentum:
    :param size:
    """
    weights, bias, loss_history, test_loss_history, accuracy_history, test_accuracy_history = optimizer(optimizer_name,
        spatial_loss_derivative, weights, bias, trainLabels, testLabels, trainImgs, testImgs,
        num_epochs, lr_weights, lr_bias, num_shots, momentum, batch_size
    )
    return weights, bias, loss_history, test_loss_history, accuracy_history, test_accuracy_history
