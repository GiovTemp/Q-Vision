# Q-Vision/qvision/qvision.py

import numpy as np
from .preprocessing import convert_to_float, convert_and_normalize, calculate_amplitudes
from .training import train


class QVision:
    def __init__(self, input_shape=(32, 32), num_epochs=150, lr_weights=0.075, lr_bias=0.005, num_shots=-1,
                 momentum=0.9, batch_size=32, ideal_conditions=True):
        self.input_shape = input_shape
        self.num_epochs = num_epochs
        self.lr_weights = lr_weights
        self.lr_bias = lr_bias
        self.num_shots = num_shots
        self.momentum = momentum
        self.batch_size = batch_size
        self.weights = None
        self.bias = 0
        self.loss_history = []
        self.test_loss_history = []
        self.accuracy_history = []
        self.test_accuracy_history = []
        self.ideal_conditions = ideal_conditions
        self.non_ideal_parameters = {
            'C': 0.0,  # iperparmetro per delta T
            'eta': 0.0,
            # vettore efficienza dei detector, ovvero rapporto fra il numero di fotoni rilevati sul numero di fotoni incidenti
            'tau': 0.0,  # vettore durata del tempo morto per i due detector (sec.)
            'drc': 0.0,  # manca nella funzione coinc
            'P': 0.0  #flusso di coppia dei fotoni espresso in Hz
        }

    def set_hyperparameters(self, num_epochs=None, lr_weights=None, lr_bias=None, num_shots=None, momentum=None,
                            batch_size=None):
        if num_epochs is not None:
            self.num_epochs = num_epochs
        if lr_weights is not None:
            self.lr_weights = lr_weights
        if lr_bias is not None:
            self.lr_bias = lr_bias
        if num_shots is not None:
            self.num_shots = num_shots
        if momentum is not None:
            self.momentum = momentum
        if batch_size is not None:
            self.batch_size = batch_size

    def set_ideal_conditions(self, ideal_conditions):
        self.ideal_conditions = ideal_conditions

    def initialize_parameters(self):
        self.weights = self.initialize_weights(self.input_shape)
        self.weights = self.normalize_weights(self.weights)
        self.bias = 0

    def update_non_ideal_parameters(self, **kwargs):
        """
        Aggiorna i parametri non ideali con i valori forniti.

        Args:
            **kwargs: Chiavi e valori per i parametri da aggiornare. Le chiavi devono
                       essere 'C', 'eta', 'tau' o 'drc'.
        """
        for key, value in kwargs.items():
            if key in self.non_ideal_parameters:
                self.non_ideal_parameters[key] = value
            else:
                print(f"Chiave '{key}' non valida. Le chiavi valide sono: {list(self.non_ideal_parameters.keys())}")

    def preprocess_data(self, train_imgs, train_labels, test_imgs, test_labels):
        #Converte in float
        train_imgs, train_labels = convert_to_float(train_imgs, train_labels)
        test_imgs, test_labels = convert_to_float(test_imgs, test_labels)
        #Converte e normalizza
        train_imgs = convert_and_normalize(train_imgs)
        test_imgs = convert_and_normalize(test_imgs)
        #Calcola le ampiezze
        train_imgs = calculate_amplitudes(train_imgs)
        test_imgs = calculate_amplitudes(test_imgs)
        return train_imgs, train_labels, test_imgs, test_labels

    def train(self, optimizer_name, train_imgs, train_labels, test_imgs, test_labels, train_source_images, train_modulated_images, train_labels1,
          test_source_images, test_modulated_images, test_labels1, phase_modulation=False):
        self.weights, self.bias, self.loss_history, self.test_loss_history, self.accuracy_history, self.test_accuracy_history = train(
            optimizer_name, self.weights, self.bias, train_imgs, train_labels, test_imgs, test_labels, self.num_epochs, train_source_images, train_modulated_images, train_labels1,
          test_source_images, test_modulated_images, test_labels1, self.lr_weights, self.lr_bias, self.num_shots, self.momentum, self.batch_size, self.ideal_conditions,
            self.non_ideal_parameters, phase_modulation)
        return self.weights, self.bias, self.loss_history, self.test_loss_history, self.accuracy_history, self.test_accuracy_history

    @staticmethod
    def initialize_weights(shape, low=0, high=1.0):
        """Initialize weights with a uniform distribution."""
        return np.random.uniform(low, high, shape)

    @staticmethod
    def normalize_weights(weights):
        """Normalize the weights."""
        norm = np.sum(np.square(weights))
        return weights / np.sqrt(norm)
