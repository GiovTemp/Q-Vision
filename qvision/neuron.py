# Q-Vision/qvision/neuron.py

import collections
import numpy as np
from typing import Callable

from .utils import sig, sigPrime, loss, accuracy
from .photon_detector import calculate_f_i


class QuantumNeuron:
    def __init__(self):
        self.N = 0  # Inizializza N solo al primo utilizzo
        self.last_f = 0

    def neuron(self, weights, bias, Img, num_shots, ideal_conditions, non_ideal_parameters):
        """ Compute the output of the quantum optical neuron, with parameters
            weights and bias, and input Img. The predicted probability is sampled
            for a given number of shots (deactived by choosing shots = -1). """

        norm = np.sqrt(np.sum(np.square(weights)))
        prob = np.abs(np.sum(np.multiply(Img, weights / norm))) ** 2

        if num_shots == -1:
            f = prob
            f_i, self.N = calculate_f_i(weights, Img, 1, ideal_conditions, non_ideal_parameters, f, self.N)
        else:
            samples = np.random.choice([0, 1], num_shots, p=[1 - prob, prob])
            counter = collections.Counter(samples)
            f = counter[1] / num_shots
            f_i, self.N = calculate_f_i(weights, Img, num_shots, ideal_conditions, non_ideal_parameters, f, self.N)

        self.last_f = f_i

        return sig(f_i + bias)

    def neuron_phase_modulation(self, weights, bias, modulated_image, num_shots):
        # _, modulated_image = gerchberg_saxton(Source, Target, max_iterations)
        modulated_image = modulated_image / np.linalg.norm(modulated_image)

        weights_phase = np.exp(2 * np.pi * 1j * weights)
        weights_phase = weights_phase / np.linalg.norm(weights_phase)

        weights2 = np.fft.fft2(weights_phase)
        weights2 = weights2 / np.linalg.norm(weights2)
        prob = np.abs(np.sum(np.multiply(modulated_image, np.conj(weights2))))**2

        if num_shots == -1:
            f = prob
        else:
            samples = np.random.choice([0, 1], num_shots, p=[1 - prob, prob])
            counter = collections.Counter(samples)
            f = counter[1]/num_shots

        return sig(f + bias, 385, 5.3)


def spatial_loss_derivative(output, target, weights, bias, Img):
    """ Compute the derivative of the binary cross-entropy with respect to the
        neuron parameters, with spatial-encoded input. """
    # Check
    if output == 1:
        raise ValueError('Output is 1!')
    elif output <= 0:
        raise ValueError('Output is negative!')
    elif 1 - output <= 0:
        raise ValueError('Output is greater than 1!')

    # Declarations
    F = output
    y = target
    norm = np.sqrt(np.sum(np.square(weights)))

    # Compute the derivative with respect to the weights
    g = np.sum(np.multiply(Img, weights / norm))  # <I, U>
    gPrime = (Img - g * weights / norm) / norm  # <I, dlambdaU>

    fPrime = 2 * np.real(g * np.conjugate(gPrime))  # 2Re[<I, U><I, dU>*]

    crossPrime = (F - y) / (F * (1 - F))

    gAbs = np.abs(g)  # sqrt(f)
    weights_derivative = crossPrime * sigPrime(gAbs ** 2 + bias) * fPrime

    # Compute the derivative with respect to the bias
    bias_derivative = crossPrime * sigPrime(gAbs ** 2 + bias)

    return weights_derivative, bias_derivative

def pm_spatial_loss_derivative(output, target, source_image, modulated_image, weights, bias):
    if output == 1:
        raise ValueError('Output is 1!')
    elif output <= 0:
        raise ValueError('Output is negative!')
    elif 1 - output <= 0:
        raise ValueError('Output is greater than 1!')

    F = output
    y = target

    # source_image, modulated_image = gerchberg_saxton(Source, Target, 1000)
    modulated_image = modulated_image / np.linalg.norm(modulated_image)

    weights_phase = np.exp(2 * np.pi * 1j * weights)
    weights_phase = weights_phase / np.linalg.norm(weights_phase)

    weights2 = np.fft.fft2(weights_phase)
    weights2 = weights2 / np.linalg.norm(weights2)

    g = np.sum(np.multiply(modulated_image, np.conj(weights2)))  # <I, U>
    gPrime = -2 * np.pi * 1j * np.multiply(source_image, np.conj(weights_phase)) * weights.size # <I, dlambdaU>

    fPrime = 2 * np.real(g * np.conjugate(gPrime))  # 2Re[<I, U><I, dU>*]

    crossPrime = (F - y) / (F * (1 - F))

    gAbs = np.abs(g)  # sqrt(f)

    weights_derivative = crossPrime * sigPrime(gAbs ** 2 + bias, 385, 5.3) * fPrime

    bias_derivative = crossPrime * sigPrime(gAbs ** 2 + bias, 385, 5.3)

    return weights_derivative, bias_derivative


def Fourier_loss_derivative(output, target, weights, bias, Img):
    """ Compute the derivative of the binary cross-entropy with respect to the
        neuron parameters, with Fourier-encoded input. """
    # Check
    if output == 1:
        raise ValueError('Output is 1!')
    elif output <= 0:
        raise ValueError('Output is negative!')
    elif 1 - output <= 0:
        raise ValueError('Output is greater than 1!')

    # Declarations
    F = output
    y = target
    norm = np.sqrt(np.sum(np.square(weights)))

    # Compute the derivative with respect to the weights
    g = np.sum(np.multiply(Img, weights / norm))  # <I, U>
    gAbs = np.abs(g)  # sqrt(f)

    gPrime = (Img - gAbs * weights / norm) / norm  # Approximation
    fPrime = 2 * np.real(gAbs * np.conjugate(gPrime))  # Approximation

    crossPrime = (F - y) / (F * (1 - F))

    weights_derivative = crossPrime * sigPrime(gAbs ** 2 + bias) * fPrime

    # Compute the derivative with respect to the bias
    bias_derivative = crossPrime * sigPrime(gAbs ** 2 + bias)

    return weights_derivative, bias_derivative


# def update_rule(weights, bias, lossWeightsDerivatives, lossBiasDerivatives, lrWeights, lrBias):
#     """ Parameters update rule of the gradient descent algorithm. """
#     new_weights = weights - lrWeights*np.mean(lossWeightsDerivatives, axis=0)
#     new_bias = bias - lrBias*np.mean(lossBiasDerivatives, axis=0)
#     return new_weights, new_bias
#
# def optimization(loss_derivative: Callable, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs, lrWeights, lrBias, num_shots):
#     """ Gradient descent optimization. """
#     # Training set
#     outputs = np.array([neuron(weights, bias, trainImgs[idx,:,:], num_shots) for idx in range(trainImgs.shape[0])])
#
#     losses = np.array([loss(outputs[idx], targets[idx]) for idx in range(outputs.shape[0])])
#
#     # History initialization
#     loss_history = [np.mean(losses)]
#     accuracy_history = [accuracy(outputs, targets)]
#
#     # Weights initialization
#     lossWeightsDerivatives = np.zeros(trainImgs.shape)
#     lossBiasDerivatives = np.zeros(trainImgs.shape[0])
#
#     # Compute derivates of the loss function
#     for idx in range(trainImgs.shape[0]):
#         lossWeightsDerivatives[idx,:,:], lossBiasDerivatives[idx] = loss_derivative(outputs[idx], targets[idx], weights, bias, trainImgs[idx,:,:])
#
#     # Validation set
#     test_outputs = np.array([neuron(weights, bias, testImgs[idx,:,:], num_shots) for idx in range(testImgs.shape[0])])
#     test_losses = np.array([loss(test_outputs[idx], test_targets[idx]) for idx in range(test_outputs.shape[0])])
#
#     test_loss_history = [np.mean(test_losses)]
#     test_accuracy_history = [accuracy(test_outputs, test_targets)]
#
#     # Verbose
#     print('EPOCH', 0)
#     print('Loss', loss_history[0], 'Val_Loss', test_loss_history[0])
#     print('Accuracy', accuracy_history[0], 'Val_Acc', test_accuracy_history[0])
#     print('---')
#
#     for epoch in range(num_epochs):
#         # Update weights
#         weights, bias = update_rule(weights, bias, lossWeightsDerivatives, lossBiasDerivatives, lrWeights, lrBias)
#
#         # Training set
#         outputs = np.array([neuron(weights, bias, trainImgs[idx,:,:], num_shots) for idx in range(trainImgs.shape[0])])
#         losses = np.array([loss(outputs[idx], targets[idx]) for idx in range(outputs.shape[0])])
#         loss_history.append(np.mean(losses))
#
#         # Update accuracy
#         accuracy_history.append(accuracy(outputs, targets))
#
#         # Validation set
#         test_outputs = np.array([neuron(weights, bias, testImgs[idx,:,:], num_shots) for idx in range(testImgs.shape[0])])
#         test_losses = np.array([loss(test_outputs[idx], test_targets[idx]) for idx in range(test_outputs.shape[0])])
#         test_loss_history.append(np.mean(test_losses))
#         test_accuracy_history.append(accuracy(test_outputs, test_targets))
#
#         # Update loss derivative
#         for idx in range(trainImgs.shape[0]):
#             lossWeightsDerivatives[idx,:,:], lossBiasDerivatives[idx] = loss_derivative(outputs[idx], targets[idx], weights, bias, trainImgs[idx,:,:])
#
#         # Verbose
#         print('EPOCH', epoch + 1)
#         print('Loss', loss_history[epoch + 1], 'Val_Loss', test_loss_history[epoch + 1])
#         print('Accuracy', accuracy_history[epoch + 1], 'Val_Acc', test_accuracy_history[epoch + 1])
#         print('---')
#
#     return weights, bias, loss_history, test_loss_history, accuracy_history, test_accuracy_history


def optimizer(optimizer_function, loss_derivative: Callable, weights, bias, targets, test_targets, trainImgs, testImgs,
              num_epochs, train_source_images, train_modulated_images, train_labels, test_source_images,
              test_modulated_images, test_labels, lrWeights, lrBias, num_shots, momentum,
              batch_size, ideal_conditions, non_ideal_parameters, phase_modulation, **kwargs):
    if optimizer_function == 'gd':
        return optimization_standard_gd(loss_derivative, weights, bias, targets, test_targets, trainImgs, testImgs,
                                        num_epochs, train_source_images, train_modulated_images, train_labels,
                                        test_source_images,
                                        test_modulated_images, test_labels, lrWeights, lrBias, num_shots,
                                        ideal_conditions,
                                        non_ideal_parameters, phase_modulation, **kwargs)
    elif optimizer_function == 'sgd':
        return optimization_sgd(loss_derivative, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs,
                                train_source_images, train_modulated_images, train_labels, test_source_images,
                                test_modulated_images, test_labels,
                                lrWeights, lrBias, num_shots, ideal_conditions, non_ideal_parameters, phase_modulation,
                                **kwargs)

    elif optimizer_function == 'sgd_momentum':
        return optimization_sgd_momentum(loss_derivative, weights, bias, targets, test_targets, trainImgs, testImgs,
                                         num_epochs, train_source_images, train_modulated_images, train_labels,
                                         test_source_images,
                                         test_modulated_images, test_labels, lrWeights, lrBias, num_shots, momentum,
                                         ideal_conditions,
                                         non_ideal_parameters, phase_modulation, **kwargs)
    elif optimizer_function == 'mini_batch_gd':
        return optimization_minibatch_gd(loss_derivative, weights, bias, targets, test_targets, trainImgs, testImgs,
                                         num_epochs, train_source_images, train_modulated_images, train_labels,
                                         test_source_images,
                                         test_modulated_images, test_labels, lrWeights, lrBias, num_shots, batch_size,
                                         ideal_conditions, non_ideal_parameters, phase_modulation, **kwargs)


def common_optimization(
        loss_derivative: Callable,
        weights: np.ndarray,
        bias: np.ndarray,
        targets: np.ndarray,
        test_targets: np.ndarray,
        trainImgs: np.ndarray,
        testImgs: np.ndarray,
        num_epochs: int,
        train_source_images: np.ndarray,
        train_modulated_images: np.ndarray,
        train_labels: np.ndarray,
        test_source_images: np.ndarray,
        test_modulated_images: np.ndarray,
        test_labels: np.ndarray,
        lrWeights: float,
        lrBias: float,
        num_shots: int,
        update_fn: Callable,
        ideal_conditions,
        non_ideal_parameters,
        phase_modulation: bool,
        **kwargs
):
    """Common optimization loop with handling for complex numbers due to phase modulation and multiple optimizers."""

    # Inizializzazione delle storie
    loss_history = []
    accuracy_history = []
    test_loss_history = []
    test_accuracy_history = []

    # Inizializzazione della cache per gli ottimizzatori che ne hanno bisogno
    cache = kwargs.get('cache', {})
    momentum = kwargs.get('momentum', 0.9)

    # Inizializzazione del modello una sola volta
    neuron_model = QuantumNeuron()

    # Determinazione della dimensione del batch e delle etichette in base alla modulazione di fase
    if phase_modulation:
        training_images_length = train_modulated_images.shape[0]
        test_images_length = test_modulated_images.shape[0]
        train_labels_used = train_labels
        test_labels_used = test_labels
        train_inputs = train_modulated_images
        test_inputs = test_modulated_images
    else:
        training_images_length = trainImgs.shape[0]
        test_images_length = testImgs.shape[0]
        train_labels_used = targets
        test_labels_used = test_targets
        train_inputs = trainImgs
        test_inputs = testImgs

    # Recupero della dimensione del batch da kwargs, default al full-batch
    batch_size = kwargs.get('batch_size', training_images_length)

    for epoch in range(num_epochs):
        # Shuffle degli indici se si utilizza mini-batch o SGD
        indices = np.arange(training_images_length)
        if batch_size < training_images_length:
            np.random.shuffle(indices)

        # Divisione in batch
        for start_idx in range(0, training_images_length, batch_size):
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]

            # Inizializzazione delle liste per accumulare i gradienti del batch
            batch_lossWeightDerivatives = []
            batch_lossBiasDerivatives = []

            # Iterazione sui campioni del batch
            for idx in batch_indices:
                # Calcolo dell'output del neurone
                if phase_modulation:
                    output = neuron_model.neuron_phase_modulation(
                        weights,
                        bias,
                        train_inputs[idx, :, :],
                        num_shots
                    )

                    # Calcolo dei derivati della loss rispetto a weights e bias
                    lossWeightDerivative, lossBiasDerivative = pm_spatial_loss_derivative(
                        np.abs(output),
                        train_labels_used[idx],
                        train_source_images[idx, :, :],
                        train_modulated_images[idx, :, :],
                        weights,
                        bias
                    )
                else:
                    output = neuron_model.neuron(
                        weights,
                        bias,
                        train_inputs[idx, :, :],
                        num_shots,
                        ideal_conditions,
                        non_ideal_parameters
                    )

                    # Calcolo dei derivati della loss rispetto a weights e bias
                    lossWeightDerivative, lossBiasDerivative = loss_derivative(
                        np.abs(output),
                        train_labels_used[idx],
                        weights,
                        bias,
                        train_inputs[idx, :, :]
                    )

                # Accumulo dei gradienti per il batch (usato per GD e mini-batch GD)
                batch_lossWeightDerivatives.append(lossWeightDerivative)
                batch_lossBiasDerivatives.append(lossBiasDerivative)

                # Aggiornamento immediato per SGD e SGD con Momentum
                if update_fn == sgd_update:
                    weights, bias, cache = update_fn(
                        weights,
                        bias,
                        lossWeightDerivative,
                        lossBiasDerivative,
                        lrWeights,
                        lrBias,
                        cache
                    )
                elif update_fn == sgd_momentum_update:
                    weights, bias, cache = update_fn(
                        weights,
                        bias,
                        lossWeightDerivative,
                        lossBiasDerivative,
                        lrWeights,
                        lrBias,
                        cache,
                        momentum
                    )

            # Dopo aver processato il batch, aggiorna per GD o mini-batch GD
            if update_fn == standard_gd_update:
                # Se è standard GD, batch_size dovrebbe essere == training_images_length
                # In caso di mini-batch GD, batch_size < training_images_length
                weights, bias, cache = update_fn(
                    weights,
                    bias,
                    batch_lossWeightDerivatives,
                    batch_lossBiasDerivatives,
                    lrWeights,
                    lrBias,
                    cache
                )

        # Calcolo delle perdite e dell'accuratezza dopo ogni epoca
        # Calcolo delle predizioni per il training set
        if phase_modulation:
            outputs = np.array([
                neuron_model.neuron_phase_modulation(weights, bias, train_inputs[idx, :, :], num_shots)
                for idx in range(training_images_length)
            ], dtype=np.complex128)
        else:
            outputs = np.array([
                neuron_model.neuron(weights, bias, train_inputs[idx, :, :], num_shots, ideal_conditions,
                                    non_ideal_parameters)
                for idx in range(training_images_length)
            ])

        # Calcolo della loss e dell'accuratezza per il training set
        losses = np.array([loss(np.abs(outputs[idx]), train_labels_used[idx]) for idx in range(outputs.shape[0])])
        loss_history.append(np.mean(losses))
        accuracy_history.append(accuracy(outputs, train_labels_used))

        # Calcolo delle predizioni per il test set
        if phase_modulation:
            test_outputs = np.array([
                neuron_model.neuron_phase_modulation(weights, bias, test_inputs[idx, :, :], num_shots)
                for idx in range(test_images_length)
            ], dtype=np.complex128)
        else:
            test_outputs = np.array([
                neuron_model.neuron(weights, bias, test_inputs[idx, :, :], num_shots, ideal_conditions,
                                    non_ideal_parameters)
                for idx in range(test_images_length)
            ])

        # Calcolo della loss e dell'accuratezza per il test set
        test_losses = np.array(
            [loss(np.abs(test_outputs[idx]), test_labels_used[idx]) for idx in range(test_outputs.shape[0])])
        test_loss_history.append(np.mean(test_losses))
        test_accuracy_history.append(accuracy(test_outputs, test_labels_used))

        # Stampa dei risultati dell'epoca corrente
        print(f'EPOCH {epoch + 1}')
        print(f'Loss: {loss_history[-1]:.4f}, Val_Loss: {test_loss_history[-1]:.4f}')
        print(f'Accuracy: {accuracy_history[-1]:.4f}, Val_Acc: {test_accuracy_history[-1]:.4f}')
        print('---')

    return weights, bias, loss_history, test_loss_history, accuracy_history, test_accuracy_history


# Define the standard gradient descent update function
def standard_gd_update(weights, bias, lossWeightsDerivatives, lossBiasDerivatives, lrWeights, lrBias, cache):
    """ Parameters update rule of the gradient descent algorithm. """
    #clipped_gradients_weights = clip_gradients(lossWeightsDerivatives)
    #clipped_gradients_bias = clip_gradients(lossBiasDerivatives)
    new_weights = weights - lrWeights * np.mean(lossWeightsDerivatives, axis=0)
    new_bias = bias - lrBias * np.mean(lossBiasDerivatives, axis=0)
    return new_weights, new_bias, cache


# Define the SGD update function
def sgd_update(weights, bias, lossWeightsDerivatives, lossBiasDerivatives, lrWeights, lrBias, cache):
    clipped_gradients_weights = clip_gradients(lossWeightsDerivatives)
    clipped_gradients_bias = clip_gradients(lossBiasDerivatives)
    weights -= lrWeights * clipped_gradients_weights
    bias -= lrBias * clipped_gradients_bias
    # weights -= lrWeights * lossWeightsDerivatives
    # bias -= lrBias * lossBiasDerivatives
    return weights, bias, cache


def sgd_momentum_update(weights, bias, lossWeightsDerivatives, lossBiasDerivatives, lrWeights, lrBias, cache,
                        momentum=0.9):
    """Update rule for Stochastic Gradient Descent (SGD) with Momentum."""

    # Ensure derivatives are treated as at least 1D arrays
    lossWeightsDerivatives = np.atleast_1d(lossWeightsDerivatives)
    lossBiasDerivatives = np.atleast_1d(lossBiasDerivatives)

    # Retrieve previous velocities from the cache, or initialize if not present
    velocity_weights = cache.get('velocity_weights', np.zeros_like(weights))
    velocity_bias = cache.get('velocity_bias', np.zeros_like(bias))

    clipped_gradients_weights = clip_gradients(lossWeightsDerivatives)
    clipped_gradients_bias = clip_gradients(lossBiasDerivatives)

    # Update velocities with the momentum term and the current gradient
    velocity_weights = momentum * velocity_weights - lrWeights * clipped_gradients_weights
    velocity_bias = momentum * velocity_bias - lrBias * clipped_gradients_bias

    # Update parameters using the velocity
    weights += velocity_weights
    bias += velocity_bias

    # Return the updated weights, bias, and cache with the updated velocities
    cache['velocity_weights'] = velocity_weights
    cache['velocity_bias'] = velocity_bias

    return weights, bias, cache

def clip_gradients(gradients, clip_value=1.0):
    gradients = np.clip(gradients, -clip_value, clip_value)
    return gradients

def optimization_standard_gd(loss_derivative, weights, bias, targets, test_targets, trainImgs, testImgs,
                             num_epochs, train_source_images, train_modulated_images, train_labels, test_source_images,
                             test_modulated_images, test_labels, lrWeights, lrBias, num_shots, ideal_conditions,
                             non_ideal_parameters, phase_modulation, **kwargs):
    return common_optimization(
        loss_derivative, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs, train_source_images,
        train_modulated_images, train_labels, test_source_images, test_modulated_images, test_labels,
        lrWeights, lrBias, num_shots, standard_gd_update, ideal_conditions, non_ideal_parameters,
        phase_modulation, **kwargs
    )


def optimization_sgd(loss_derivative, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs,
                     train_source_images, train_modulated_images, train_labels, test_source_images,
                     test_modulated_images, test_labels,
                     lrWeights, lrBias, num_shots, ideal_conditions, non_ideal_parameters, phase_modulation, **kwargs):
    kwargs['batch_size'] = 1

    return common_optimization(
        loss_derivative, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs, train_source_images,
        train_modulated_images, train_labels, test_source_images, test_modulated_images, test_labels,
        lrWeights, lrBias, num_shots, sgd_update, ideal_conditions, non_ideal_parameters, phase_modulation, **kwargs
    )


def optimization_sgd_momentum(loss_derivative, weights, bias, targets, test_targets, trainImgs, testImgs,
                              num_epochs, train_source_images, train_modulated_images, train_labels, test_source_images,
                              test_modulated_images, test_labels, lrWeights, lrBias, num_shots, momentum,
                              ideal_conditions,
                              non_ideal_parameters, phase_modulation, **kwargs):
    kwargs['batch_size'] = 1
    kwargs['momentum'] = momentum

    return common_optimization(
        loss_derivative, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs,
        train_source_images, train_modulated_images, train_labels, test_source_images, test_modulated_images,
        test_labels,
        lrWeights, lrBias, num_shots, sgd_momentum_update, ideal_conditions, non_ideal_parameters,
        phase_modulation, **kwargs
    )


def optimization_minibatch_gd(loss_derivative, weights, bias, targets, test_targets, trainImgs, testImgs,
                              num_epochs, train_source_images, train_modulated_images, train_labels, test_source_images,
                              test_modulated_images, test_labels, lrWeights, lrBias, num_shots, batch_size,
                              ideal_conditions, non_ideal_parameters, phase_modulation, **kwargs):
    kwargs['batch_size'] = batch_size

    return common_optimization(
        loss_derivative, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs, train_source_images,
        train_modulated_images, train_labels, test_source_images,
        test_modulated_images, test_labels,
        lrWeights, lrBias, num_shots, standard_gd_update, ideal_conditions, non_ideal_parameters, phase_modulation,
        **kwargs
    )
