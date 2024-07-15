# Q-Vision/qvision/training.py

import numpy as np
from .neuron import neuron, optimization_rmsprop, spatial_loss_derivative
from .utils import loss, accuracy

def train(weights, bias, trainImgs, trainLabels, testImgs, testLabels, num_epochs, lr_weights, lr_bias, num_shots):
    """Train the model using the optimization function from neuron.py"""
    weights, bias, loss_history, test_loss_history, accuracy_history, test_accuracy_history = optimization(
        spatial_loss_derivative, weights, bias, trainLabels, testLabels, trainImgs, testImgs,
        num_epochs, lr_weights, lr_bias, num_shots
    )
    return weights, bias, loss_history, test_loss_history, accuracy_history, test_accuracy_history
