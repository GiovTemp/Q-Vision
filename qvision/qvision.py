# Q-Vision/qvision/qvision.py

import numpy as np
from .preprocessing import convert_to_float, convert_and_normalize, calculate_amplitudes
from .training import train

class QVision:
    def __init__(self, input_shape=(32, 32), num_epochs=150, lr_weights=0.075, lr_bias=0.005, num_shots=-1):
        self.input_shape = input_shape
        self.num_epochs = num_epochs
        self.lr_weights = lr_weights
        self.lr_bias = lr_bias
        self.num_shots = num_shots
        self.weights = None
        self.bias = 0
        self.loss_history = []
        self.test_loss_history = []
        self.accuracy_history = []
        self.test_accuracy_history = []

    def set_hyperparameters(self, num_epochs=None, lr_weights=None, lr_bias=None, num_shots=None):
        if num_epochs is not None:
            self.num_epochs = num_epochs
        if lr_weights is not None:
            self.lr_weights = lr_weights
        if lr_bias is not None:
            self.lr_bias = lr_bias
        if num_shots is not None:
            self.num_shots = num_shots

    def initialize_parameters(self):
        self.weights = self.initialize_weights(self.input_shape)
        self.weights = self.normalize_weights(self.weights)
        self.bias = 0

    def preprocess_data(self, train_imgs, train_labels, test_imgs, test_labels):
        train_imgs, train_labels = convert_to_float(train_imgs, train_labels)
        test_imgs, test_labels = convert_to_float(test_imgs, test_labels)
        train_imgs = convert_and_normalize(train_imgs)
        test_imgs = convert_and_normalize(test_imgs)
        train_imgs = calculate_amplitudes(train_imgs)
        test_imgs = calculate_amplitudes(test_imgs)
        return train_imgs, train_labels, test_imgs, test_labels

    def train(self, optimizer_name, train_imgs, train_labels, test_imgs, test_labels):
        self.weights, self.bias, self.loss_history, self.test_loss_history, self.accuracy_history, self.test_accuracy_history = train(
            optimizer_name, self.weights, self.bias, train_imgs, train_labels, test_imgs, test_labels,
            self.num_epochs, self.lr_weights, self.lr_bias, self.num_shots
        )
        return self.weights, self.bias, self.loss_history, self.test_loss_history, self.accuracy_history, self.test_accuracy_history

    @staticmethod
    def initialize_weights(shape, low=-1.0, high=1.0):
        """Initialize weights with a uniform distribution."""
        return np.random.uniform(low, high, shape)

    @staticmethod
    def normalize_weights(weights):
        """Normalize the weights."""
        norm = np.sum(np.square(weights))
        return weights / np.sqrt(norm)