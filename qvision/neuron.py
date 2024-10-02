# Q-Vision/qvision/neuron.py

import collections
import numpy as np
from typing import Callable, Tuple

from .utils import sig, sigPrime, loss, accuracy

def neuron(weights, bias, Img, num_shots):
    """ Compute the output of the quantum optical neuron, with parameters
        weights and bias, and input Img. The predicted probability is sampled
        for a given number of shots (deactived by choosing shots = -1). """
    norm = np.sqrt(np.sum(np.square(weights)))
    prob = np.abs(np.sum(np.multiply(Img, weights/norm)))**2
    # Sampling (1: Coincidence)
    if num_shots == -1:
        f = prob
    else:
        samples = np.random.choice([0, 1], num_shots, p=[1 - prob, prob])
        counter = collections.Counter(samples)
        f = counter[1]/num_shots
    return sig(f + bias)

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
    g = np.sum(np.multiply(Img, weights/norm)) # <I, U>
    gPrime = (Img - g*weights/norm)/norm # <I, dlambdaU>

    fPrime = 2*np.real(g*np.conjugate(gPrime)) # 2Re[<I, U><I, dU>*]

    crossPrime = (F - y)/(F*(1-F))

    gAbs = np.abs(g) # sqrt(f)
    weights_derivative = crossPrime*sigPrime(gAbs**2 + bias)*fPrime

    # Compute the derivative with respect to the bias
    bias_derivative = crossPrime*sigPrime(gAbs**2 + bias)

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
    g = np.sum(np.multiply(Img, weights/norm)) # <I, U>
    gAbs = np.abs(g) # sqrt(f)

    gPrime = (Img - gAbs*weights/norm)/norm # Approximation
    fPrime = 2*np.real(gAbs*np.conjugate(gPrime)) # Approximation

    crossPrime = (F - y)/(F*(1-F))

    weights_derivative = crossPrime*sigPrime(gAbs**2 + bias)*fPrime

    # Compute the derivative with respect to the bias
    bias_derivative = crossPrime*sigPrime(gAbs**2 + bias)

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
              num_epochs, lrWeights, lrBias, num_shots, momentum, batch_size, **kwargs):
    if optimizer_function == 'gd':
        return optimization_standard_gd(loss_derivative, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs, lrWeights, lrBias, num_shots, **kwargs)
    elif optimizer_function == 'sgd':
        return optimization_sgd(loss_derivative, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs, lrWeights, lrBias, num_shots, **kwargs)
    elif optimizer_function == 'sgd_momentum':
        return optimization_sgd_momentum(loss_derivative, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs, lrWeights, lrBias, num_shots, momentum, **kwargs)
    elif optimizer_function == 'mini_batch_gd':
        return optimization_minibatch_gd(loss_derivative, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs, lrWeights, lrBias, num_shots, batch_size, **kwargs)

def common_optimization(
    loss_derivative: Callable, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs,
    lrWeights, lrBias, num_shots, update_fn: Callable, **kwargs
):
    """Common optimization loop."""
    # History initialization
    loss_history = []
    accuracy_history = []

    # Cache initialization
    cache = kwargs.get('cache', {})
    momentum = kwargs.get('momentum', 0.9)

    # Recupero la dimensione del batch dai kwargs, default 1 per SGD e 32 per Mini-Batch GD
    batch_size = kwargs.get('batch_size', 1)  # Default a 1 per SGD

    # Verbose initial values
    print('EPOCH', 0)
    initial_outputs = np.array([neuron(weights, bias, trainImgs[idx, :, :], num_shots) for idx in range(trainImgs.shape[0])])
    initial_losses = np.array([loss(initial_outputs[idx], targets[idx]) for idx in range(initial_outputs.shape[0])])
    initial_test_outputs = np.array([neuron(weights, bias, testImgs[idx, :, :], num_shots) for idx in range(testImgs.shape[0])])
    initial_test_losses = np.array([loss(initial_test_outputs[idx], test_targets[idx]) for idx in range(initial_test_outputs.shape[0])])

    loss_history.append(np.mean(initial_losses))
    accuracy_history.append(accuracy(initial_outputs, targets))

    test_loss_history = [np.mean(initial_test_losses)]
    test_accuracy_history = [accuracy(initial_test_outputs, test_targets)]

    print('Loss', loss_history[0], 'Val_Loss', test_loss_history[0])
    print('Accuracy', accuracy_history[0], 'Val_Acc', test_accuracy_history[0])
    print('---')

    for epoch in range(num_epochs):
        if update_fn == standard_gd_update:
            # No shuffle for standard GD
            outputs = np.array([neuron(weights, bias, trainImgs[idx, :, :], num_shots) for idx in range(trainImgs.shape[0])])
            losses = np.array([loss(outputs[idx], targets[idx]) for idx in range(outputs.shape[0])])
            lossWeightsDerivatives = np.array([np.atleast_1d(loss_derivative(outputs[idx], targets[idx], weights, bias, trainImgs[idx, :, :])[0]) for idx in range(trainImgs.shape[0])])
            lossBiasDerivatives = np.array([np.atleast_1d(loss_derivative(outputs[idx], targets[idx], weights, bias, trainImgs[idx, :, :])[1]) for idx in range(trainImgs.shape[0])])

            mean_lossWeightsDerivatives = np.mean(lossWeightsDerivatives, axis=0)
            mean_lossBiasDerivatives = np.mean(lossBiasDerivatives, axis=0)

            # Update weights once per epoch using all data
            weights, bias, cache = update_fn(
                weights, bias, mean_lossWeightsDerivatives, mean_lossBiasDerivatives, lrWeights, lrBias, cache
            )

        else:
            # Shuffle the dataset at the start of each epoch (only for SGD or Mini-Batch)
            indices = np.random.permutation(trainImgs.shape[0])

            # Iterate over mini-batches or individual samples if batch_size = 1
            for start_idx in range(0, trainImgs.shape[0], batch_size):
                end_idx = min(start_idx + batch_size, trainImgs.shape[0])
                batch_indices = indices[start_idx:end_idx]

                batch_lossWeightDerivatives = []
                batch_lossBiasDerivatives = []

                # Iterate over samples in the current batch
                for idx in batch_indices:
                    output = neuron(weights, bias, trainImgs[idx, :, :], num_shots)
                    lossWeightDerivative, lossBiasDerivative = loss_derivative(output, targets[idx], weights, bias, trainImgs[idx, :, :])

                    batch_lossWeightDerivatives.append(lossWeightDerivative)
                    batch_lossBiasDerivatives.append(lossBiasDerivative)

                # Calcola la media dei gradienti solo per Mini-Batch GD (batch_size > 1)
                if batch_size > 1:  # Mini-Batch GD or Standard GD
                    mean_lossWeightDerivatives = np.mean(np.atleast_1d(batch_lossWeightDerivatives), axis=0)
                    mean_lossBiasDerivatives = np.mean(np.atleast_1d(batch_lossBiasDerivatives), axis=0)
                else:  # Stochastic GD (batch_size == 1)
                    mean_lossWeightDerivatives = batch_lossWeightDerivatives[0]
                    mean_lossBiasDerivatives = batch_lossBiasDerivatives[0]

                # Filtra il batch_size e momentum da kwargs prima di chiamare la funzione di aggiornamento
                filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['batch_size', 'momentum']}

                # Update weights depending on the update function (SGD, momentum, or standard GD)
                if update_fn == sgd_momentum_update:
                    weights, bias, cache = update_fn(
                        weights, bias, mean_lossWeightDerivatives, mean_lossBiasDerivatives, lrWeights, lrBias, cache, momentum=momentum, **filtered_kwargs
                    )
                else:
                    weights, bias, cache = update_fn(
                        weights, bias, mean_lossWeightDerivatives, mean_lossBiasDerivatives, lrWeights, lrBias, cache, **filtered_kwargs
                    )

        # After each epoch, calculate losses and accuracy
        outputs = np.array([neuron(weights, bias, trainImgs[idx, :, :], num_shots) for idx in range(trainImgs.shape[0])])
        losses = np.array([loss(outputs[idx], targets[idx]) for idx in range(outputs.shape[0])])
        loss_history.append(np.mean(losses))

        accuracy_history.append(accuracy(outputs, targets))

        test_outputs = np.array([neuron(weights, bias, testImgs[idx, :, :], num_shots) for idx in range(testImgs.shape[0])])
        test_losses = np.array([loss(test_outputs[idx], test_targets[idx]) for idx in range(test_outputs.shape[0])])
        test_loss_history.append(np.mean(test_losses))
        test_accuracy_history.append(accuracy(test_outputs, test_targets))

        print('EPOCH', epoch + 1)
        print('Loss', loss_history[epoch + 1], 'Val_Loss', test_loss_history[epoch + 1])
        print('Accuracy', accuracy_history[epoch + 1], 'Val_Acc', test_accuracy_history[epoch + 1])
        print('---')

    return weights, bias, loss_history, test_loss_history, accuracy_history, test_accuracy_history
# Define the standard gradient descent update function
def standard_gd_update(weights, bias, lossWeightsDerivatives, lossBiasDerivatives, lrWeights, lrBias, cache):
    """ Parameters update rule of the gradient descent algorithm. """
    new_weights = weights - lrWeights * np.mean(lossWeightsDerivatives, axis=0)
    new_bias = bias - lrBias * np.mean(lossBiasDerivatives, axis=0)
    return new_weights, new_bias, cache


# Define the SGD update function
def sgd_update(weights, bias, lossWeightsDerivatives, lossBiasDerivatives, lrWeights, lrBias, cache):
    weights -= lrWeights * lossWeightsDerivatives
    bias -= lrBias * lossBiasDerivatives
    return weights, bias, cache

# def minibatch_gd_update(weights, bias, lossWeightsDerivatives, lossBiasDerivatives, lrWeights, lrBias, cache, batch_size):
#     """ Update rule for Mini-Batch Gradient Descent (MBGD) """
#     # Qui calcoliamo la media su ciascun mini-batch
#     new_weights = weights - lrWeights * np.mean(lossWeightsDerivatives, axis=0)
#     new_bias = bias - lrBias * np.mean(lossBiasDerivatives, axis=0)
#     return new_weights, new_bias, cache

def sgd_momentum_update(weights, bias, lossWeightsDerivatives, lossBiasDerivatives, lrWeights, lrBias, cache, momentum=0.9):
    """ Update rule for Stochastic Gradient Descent (SGD) with Momentum """
    # Assicurati che le derivate siano trattate come array di almeno 1D
    lossWeightsDerivatives = np.atleast_1d(lossWeightsDerivatives)
    lossBiasDerivatives = np.atleast_1d(lossBiasDerivatives)

    # Recupera le velocità dai valori precedenti
    velocity_weights = cache.get('velocity_weights', np.zeros_like(weights))
    velocity_bias = cache.get('velocity_bias', np.zeros_like(bias))


    # Aggiorna le velocità con il termine di momentum e il fattore (1 - momentum)
    velocity_weights = momentum * velocity_weights + (1 - momentum) * lossWeightsDerivatives
    velocity_bias = momentum * velocity_bias + (1 - momentum) * lossBiasDerivatives

    # Aggiorna i parametri (pesi e bias)
    weights -= velocity_weights
    bias -= velocity_bias

    # Ritorna i pesi aggiornati, il bias aggiornato, e le velocità
    return weights, bias, {'velocity_weights': velocity_weights, 'velocity_bias': velocity_bias}

def optimization_standard_gd(
        loss_derivative: Callable, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs,
        lrWeights, lrBias, num_shots, **kwargs
):
    return common_optimization(
        loss_derivative, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs,
        lrWeights, lrBias, num_shots, standard_gd_update, **kwargs
    )

def optimization_sgd(
        loss_derivative: Callable, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs,
        lrWeights, lrBias, num_shots, **kwargs
):
    return common_optimization(
        loss_derivative, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs,
        lrWeights, lrBias, num_shots, sgd_update, batch_size=1, **kwargs
    )

def optimization_sgd_momentum(
        loss_derivative: Callable, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs,
        lrWeights, lrBias, num_shots, momentum, **kwargs
):
    return common_optimization(
        loss_derivative, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs,
        lrWeights, lrBias, num_shots, sgd_momentum_update, momentum=momentum, batch_size=1, **kwargs
    )

def optimization_minibatch_gd(
        loss_derivative: Callable, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs,
        lrWeights, lrBias, num_shots, batch_size, **kwargs
):
    return common_optimization(
        loss_derivative, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs,
        lrWeights, lrBias, num_shots, standard_gd_update, batch_size=batch_size, **kwargs
    )


