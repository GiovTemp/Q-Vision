# Q-Vision/qvision/training.py

import numpy as np
from .neuron import optimizer, spatial_loss_derivative, pm_spatial_loss_derivative
from .utils import loss, accuracy

def train(optimizer_name, weights, bias, trainImgs, trainLabels, testImgs, testLabels, num_epochs, train_source_images, train_modulated_images, train_labels1,
          test_source_images, test_modulated_images, test_labels1, lr_weights, lr_bias,
          num_shots, momentum, batch_size, ideal_conditions, non_ideal_parameters, phase_modulation):
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

    loss_derivative = pm_spatial_loss_derivative if phase_modulation else spatial_loss_derivative

    weights, bias, loss_history, test_loss_history, accuracy_history, test_accuracy_history = optimizer(optimizer_name,
        loss_derivative, weights, bias, trainLabels, testLabels, trainImgs, testImgs,
        num_epochs, train_source_images, train_modulated_images, train_labels1,
          test_source_images, test_modulated_images, test_labels1, lr_weights, lr_bias, num_shots, momentum, batch_size, ideal_conditions, non_ideal_parameters, phase_modulation
    )

    return weights, bias, loss_history, test_loss_history, accuracy_history, test_accuracy_history
