# Q-Vision/qvision/training.py

import numpy as np

from .neuron import neuron, optimizer, spatial_loss_derivative
from .phase_modulation import spatial_loss_derivative as pm_spatial_loss_derivative
from .utils import loss, accuracy

def train(optimizer_name, weights, bias, trainImgs, trainLabels, testImgs, testLabels, train_source_images, train_modulated_images, train_labels,
          test_source_images, test_modulated_images, test_labels, num_epochs, lr_weights, lr_bias,
          num_shots, phase_modulation, momentum, batch_size):
    """Train the model using the optimization function from neuron.py
    :param momentum:
    :param size:
    """

    if phase_modulation:
        weights, bias, loss_history, test_loss_history, accuracy_history, test_accuracy_history = optimizer(optimizer_name,
            pm_spatial_loss_derivative, weights, bias, trainLabels, testLabels, trainImgs, testImgs, train_source_images, train_modulated_images, train_labels,
          test_source_images, test_modulated_images, test_labels, num_epochs, lr_weights, lr_bias, num_shots, momentum, batch_size, phase_modulation)
    else:
        weights, bias, loss_history, test_loss_history, accuracy_history, test_accuracy_history = optimizer(optimizer_name,
            spatial_loss_derivative, weights, bias, trainLabels, testLabels, trainImgs, testImgs, train_source_images, train_modulated_images, train_labels,
          test_source_images, test_modulated_images, test_labels, num_epochs, lr_weights, lr_bias, num_shots, momentum, batch_size, phase_modulation)

    return weights, bias, loss_history, test_loss_history, accuracy_history, test_accuracy_history
