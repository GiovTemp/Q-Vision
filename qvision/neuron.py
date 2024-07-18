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

def optimizer(optimizer, loss_derivative: Callable, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs, lrWeights, lrBias, num_shots):
    if optimizer == 'gd':
        return optimization_standard_gd(loss_derivative, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs,lrWeights, lrBias, num_shots)
    if optimizer == 'rmsprop':
         return optimization_rmsprop(loss_derivative, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs,lrWeights, lrBias, num_shots, decay_rate=0.9, epsilon=1e-8)
    elif optimizer == 'adam':
        return optimization_adam(loss_derivative, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs, lrWeights, lrBias, num_shots, beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif optimizer == 'sgd':
        return optimization_sgd_momentum(loss_derivative, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs, lrWeights, lrBias, num_shots, momentum=0.9)
    elif optimizer == 'sgd_momentum':
        return optimization_sgd(loss_derivative, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs, lrWeights, lrBias, num_shots)
    elif optimizer == 'ada_grad':
        return optimization_adagrad(loss_derivative, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs, lrWeights, lrBias, num_shots, epsilon=1e-8)
    elif optimizer == 'rmsprop_momentum':
        return optimization_sgd_momentum(loss_derivative, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs, lrWeights, lrBias, num_shots, momentum=0.9)
    elif optimizer == 'ada_delta':
        return optimization_adadelta(loss_derivative, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs, lrWeights, lrBias, num_shots, epsilon=1e-8, rho=0.9)


# Define the common optimization function
def common_optimization(
        loss_derivative: Callable, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs,
        lrWeights, lrBias, num_shots, update_fn: Callable, **kwargs
):
    """Common optimization loop."""
    # Training set
    outputs = np.array([neuron(weights, bias, trainImgs[idx, :, :], num_shots) for idx in range(trainImgs.shape[0])])
    losses = np.array([loss(outputs[idx], targets[idx]) for idx in range(outputs.shape[0])])

    # History initialization
    loss_history = [np.mean(losses)]
    accuracy_history = [accuracy(outputs, targets)]

    # Weights initialization
    lossWeightsDerivatives = np.zeros(trainImgs.shape)
    lossBiasDerivatives = np.zeros(trainImgs.shape[0])

    # Cache initialization
    cache = kwargs.get('cache', {})
    t = kwargs.get('t', 1)

    # Compute derivatives of the loss function
    for idx in range(trainImgs.shape[0]):
        lossWeightsDerivatives[idx, :, :], lossBiasDerivatives[idx] = loss_derivative(
            outputs[idx], targets[idx], weights, bias, trainImgs[idx, :, :]
        )

    # Validation set
    test_outputs = np.array([neuron(weights, bias, testImgs[idx, :, :], num_shots) for idx in range(testImgs.shape[0])])
    test_losses = np.array([loss(test_outputs[idx], test_targets[idx]) for idx in range(test_outputs.shape[0])])

    test_loss_history = [np.mean(test_losses)]
    test_accuracy_history = [accuracy(test_outputs, test_targets)]

    # Verbose
    print('EPOCH', 0)
    print('Loss', loss_history[0], 'Val_Loss', test_loss_history[0])
    print('Accuracy', accuracy_history[0], 'Val_Acc', test_accuracy_history[0])
    print('---')

    for epoch in range(num_epochs):
        # Update weights using the specific optimizer
        weights, bias, cache = update_fn(
            weights, bias, lossWeightsDerivatives, lossBiasDerivatives, lrWeights, lrBias, cache, **kwargs
        )

        # Training set
        outputs = np.array(
            [neuron(weights, bias, trainImgs[idx, :, :], num_shots) for idx in range(trainImgs.shape[0])])
        losses = np.array([loss(outputs[idx], targets[idx]) for idx in range(outputs.shape[0])])
        loss_history.append(np.mean(losses))

        # Update accuracy
        accuracy_history.append(accuracy(outputs, targets))

        # Validation set
        test_outputs = np.array(
            [neuron(weights, bias, testImgs[idx, :, :], num_shots) for idx in range(testImgs.shape[0])])
        test_losses = np.array([loss(test_outputs[idx], test_targets[idx]) for idx in range(test_outputs.shape[0])])
        test_loss_history.append(np.mean(test_losses))
        test_accuracy_history.append(accuracy(test_outputs, test_targets))

        # Update loss derivative
        for idx in range(trainImgs.shape[0]):
            lossWeightsDerivatives[idx, :, :], lossBiasDerivatives[idx] = loss_derivative(
                outputs[idx], targets[idx], weights, bias, trainImgs[idx, :, :]
            )

        # Verbose
        print('EPOCH', epoch + 1)
        print('Loss', loss_history[epoch + 1], 'Val_Loss', test_loss_history[epoch + 1])
        print('Accuracy', accuracy_history[epoch + 1], 'Val_Acc', test_accuracy_history[epoch + 1])
        print('---')

    return weights, bias, loss_history, test_loss_history, accuracy_history, test_accuracy_history

def standard_gd_update(weights, bias, lossWeightsDerivatives, lossBiasDerivatives, lrWeights, lrBias, cache):
    """ Parameters update rule of the gradient descent algorithm. """
    new_weights = weights - lrWeights * np.mean(lossWeightsDerivatives, axis=0)
    new_bias = bias - lrBias * np.mean(lossBiasDerivatives, axis=0)
    return new_weights, new_bias, cache

# Define the AdaDelta update function
def adadelta_update(weights, bias, lossWeightsDerivatives, lossBiasDerivatives, lrWeights, lrBias, cache, epsilon=1e-8, rho=0.9):
    # Initialize cache if not already initialized
    if 'accumulated_gradient_weights' not in cache:
        cache['accumulated_gradient_weights'] = np.zeros_like(weights)
    if 'accumulated_gradient_bias' not in cache:
        cache['accumulated_gradient_bias'] = np.zeros_like(bias)
    if 'accumulated_update_weights' not in cache:
        cache['accumulated_update_weights'] = np.zeros_like(weights)
    if 'accumulated_update_bias' not in cache:
        cache['accumulated_update_bias'] = np.zeros_like(bias)

    # Compute RMS of gradients
    accumulated_gradient_weights = cache['accumulated_gradient_weights']
    accumulated_gradient_bias = cache['accumulated_gradient_bias']

    accumulated_gradient_weights = rho * accumulated_gradient_weights + (1 - rho) * np.square(lossWeightsDerivatives)
    accumulated_gradient_bias = rho * accumulated_gradient_bias + (1 - rho) * np.square(lossBiasDerivatives)

    # Compute update step
    update_weights = np.sqrt((cache['accumulated_update_weights'] + epsilon) / (accumulated_gradient_weights + epsilon)) * lossWeightsDerivatives
    update_bias = np.sqrt((cache['accumulated_update_bias'] + epsilon) / (accumulated_gradient_bias + epsilon)) * lossBiasDerivatives

    # Update accumulated update step
    cache['accumulated_update_weights'] = rho * cache['accumulated_update_weights'] + (1 - rho) * np.square(update_weights)
    cache['accumulated_update_bias'] = rho * cache['accumulated_update_bias'] + (1 - rho) * np.square(update_bias)

    # Update weights and bias
    weights -= lrWeights * update_weights
    bias -= lrBias * update_bias

    # Update accumulated gradient
    cache['accumulated_gradient_weights'] = accumulated_gradient_weights
    cache['accumulated_gradient_bias'] = accumulated_gradient_bias

    return weights, bias, cache

# Define the RMSProp with momentum update function
def rmsprop_momentum_update(weights, bias, lossWeightsDerivatives, lossBiasDerivatives, lrWeights, lrBias, cache,
                            decay_rate=0.9, epsilon=1e-8, momentum=0.9):
    # Initialize cache if not already initialized
    if 'cache_weights' not in cache:
        cache['cache_weights'] = np.zeros_like(weights)
    if 'cache_bias' not in cache:
        cache['cache_bias'] = np.zeros_like(bias)
    if 'velocity_weights' not in cache:
        cache['velocity_weights'] = np.zeros_like(weights)
    if 'velocity_bias' not in cache:
        cache['velocity_bias'] = np.zeros_like(bias)

    # Update cache with squared gradients
    cache['cache_weights'] = decay_rate * cache['cache_weights'] + (1 - decay_rate) * np.square(lossWeightsDerivatives)
    cache['cache_bias'] = decay_rate * cache['cache_bias'] + (1 - decay_rate) * np.square(lossBiasDerivatives)

    # Update velocity with momentum
    cache['velocity_weights'] = momentum * cache['velocity_weights'] + lrWeights * lossWeightsDerivatives / (
                np.sqrt(cache['cache_weights']) + epsilon)
    cache['velocity_bias'] = momentum * cache['velocity_bias'] + lrBias * lossBiasDerivatives / (
                np.sqrt(cache['cache_bias']) + epsilon)

    # Update weights and bias
    weights -= cache['velocity_weights']
    bias -= cache['velocity_bias']

    return weights, bias, cache

# Define the AdaGrad update function
def adagrad_update(weights, bias, lossWeightsDerivatives, lossBiasDerivatives, lrWeights, lrBias, cache, epsilon=1e-8):
    # Initialize cache if not already initialized
    if 'cache_weights' not in cache:
        cache['cache_weights'] = np.zeros_like(weights)
    if 'cache_bias' not in cache:
        cache['cache_bias'] = np.zeros_like(bias)

    # Update cache
    cache['cache_weights'] += np.square(lossWeightsDerivatives)
    cache['cache_bias'] += np.square(lossBiasDerivatives)

    # Compute updates
    weights -= lrWeights * lossWeightsDerivatives / (np.sqrt(cache['cache_weights']) + epsilon)
    bias -= lrBias * lossBiasDerivatives / (np.sqrt(cache['cache_bias']) + epsilon)

    return weights, bias, cache

# Define the SGD update function
def sgd_update(weights, bias, lossWeightsDerivatives, lossBiasDerivatives, lrWeights, lrBias, cache):
    # Update parameters
    weights -= lrWeights * np.mean(lossWeightsDerivatives, axis=0)
    bias -= lrBias * np.mean(lossBiasDerivatives, axis=0)

    return weights, bias, cache

# Define the SGD with momentum update function
def sgd_momentum_update(weights, bias, lossWeightsDerivatives, lossBiasDerivatives, lrWeights, lrBias, cache, momentum=0.9):
    velocity_weights = cache.get('velocity_weights', np.zeros_like(weights))
    velocity_bias = cache.get('velocity_bias', np.zeros_like(bias))

    # Update velocities
    velocity_weights = momentum * velocity_weights + lrWeights * np.mean(lossWeightsDerivatives, axis=0)
    velocity_bias = momentum * velocity_bias + lrBias * np.mean(lossBiasDerivatives, axis=0)

    # Update parameters
    weights -= velocity_weights
    bias -= velocity_bias

    return weights, bias, {'velocity_weights': velocity_weights, 'velocity_bias': velocity_bias}


# Define the RMSProp update function
def rmsprop_update(weights, bias, lossWeightsDerivatives, lossBiasDerivatives, lrWeights, lrBias, cache,
                   decay_rate=0.9, epsilon=1e-8):
    cache_weights, cache_bias = cache.get('weights', np.zeros_like(weights)), cache.get('bias', np.zeros_like(bias))

    cache_weights = decay_rate * cache_weights + (1 - decay_rate) * np.mean(np.square(lossWeightsDerivatives), axis=0)
    cache_bias = decay_rate * cache_bias + (1 - decay_rate) * np.mean(np.square(lossBiasDerivatives), axis=0)

    weights -= lrWeights * np.mean(lossWeightsDerivatives, axis=0) / (np.sqrt(cache_weights) + epsilon)
    bias -= lrBias * np.mean(lossBiasDerivatives, axis=0) / (np.sqrt(cache_bias) + epsilon)

    return weights, bias, {'weights': cache_weights, 'bias': cache_bias}


# Define the Adam update function
def adam_update(weights, bias, lossWeightsDerivatives, lossBiasDerivatives, lrWeights, lrBias, cache, t=1, beta1=0.9,
                beta2=0.999, epsilon=1e-8):
    m_weights = cache.get('m_weights', np.zeros_like(weights))
    v_weights = cache.get('v_weights', np.zeros_like(weights))
    m_bias = cache.get('m_bias', np.zeros_like(bias))
    v_bias = cache.get('v_bias', np.zeros_like(bias))

    # Update biased first moment estimate
    m_weights = beta1 * m_weights + (1 - beta1) * np.mean(lossWeightsDerivatives, axis=0)
    m_bias = beta1 * m_bias + (1 - beta1) * np.mean(lossBiasDerivatives, axis=0)

    # Update biased second raw moment estimate
    v_weights = beta2 * v_weights + (1 - beta2) * np.mean(np.square(lossWeightsDerivatives), axis=0)
    v_bias = beta2 * v_bias + (1 - beta2) * np.mean(np.square(lossBiasDerivatives), axis=0)

    # Compute bias-corrected first moment estimate
    m_hat_weights = m_weights / (1 - beta1 ** t)
    m_hat_bias = m_bias / (1 - beta1 ** t)

    # Compute bias-corrected second raw moment estimate
    v_hat_weights = v_weights / (1 - beta2 ** t)
    v_hat_bias = v_bias / (1 - beta2 ** t)

    # Update parameters
    weights -= lrWeights * m_hat_weights / (np.sqrt(v_hat_weights) + epsilon)
    bias -= lrBias * m_hat_bias / (np.sqrt(v_hat_bias) + epsilon)

    return weights, bias, {'m_weights': m_weights, 'v_weights': v_weights, 'm_bias': m_bias, 'v_bias': v_bias}

def optimization_standard_gd(
        loss_derivative: Callable, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs,
        lrWeights, lrBias, num_shots
):
    return common_optimization(
        loss_derivative, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs,
        lrWeights, lrBias, num_shots, standard_gd_update
    )


# RMSProp optimization function
def optimization_rmsprop(
        loss_derivative: Callable, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs,
        lrWeights, lrBias, num_shots, decay_rate=0.9, epsilon=1e-8
):
    return common_optimization(
        loss_derivative, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs,
        lrWeights, lrBias, num_shots, rmsprop_update, decay_rate=decay_rate, epsilon=epsilon
    )


# Adam optimization function
def optimization_adam(
        loss_derivative: Callable, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs,
        lrWeights, lrBias, num_shots, beta1=0.9, beta2=0.999, epsilon=1e-8
):
    return common_optimization(
        loss_derivative, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs,
        lrWeights, lrBias, num_shots, adam_update, beta1=beta1, beta2=beta2, epsilon=epsilon, t=1
    )

# SGD with momentum optimization function
def optimization_sgd_momentum(
    loss_derivative: Callable, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs,
    lrWeights, lrBias, num_shots, momentum=0.9
):
    return common_optimization(
        loss_derivative, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs,
        lrWeights, lrBias, num_shots, sgd_momentum_update, momentum=momentum
    )

# SGD optimization function
def optimization_sgd(
    loss_derivative: Callable, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs,
    lrWeights, lrBias, num_shots
):
    return common_optimization(
        loss_derivative, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs,
        lrWeights, lrBias, num_shots, sgd_update
    )

# AdaGrad optimization function
def optimization_adagrad(
    loss_derivative: Callable, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs,
    lrWeights, lrBias, num_shots, epsilon=1e-8
):
    return common_optimization(
        loss_derivative, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs,
        lrWeights, lrBias, num_shots, adagrad_update, epsilon=epsilon
    )

# RMSProp with momentum optimization function
def optimization_rmsprop_momentum(
        loss_derivative: Callable, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs,
        lrWeights, lrBias, num_shots, decay_rate=0.9, epsilon=1e-8, momentum=0.9
):
    return common_optimization(
        loss_derivative, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs,
        lrWeights, lrBias, num_shots, rmsprop_momentum_update, decay_rate=decay_rate, epsilon=epsilon, momentum=momentum
    )

# AdaDelta optimization function
def optimization_adadelta(
        loss_derivative: Callable, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs,
        lrWeights, lrBias, num_shots, epsilon=1e-8, rho=0.9
):
    return common_optimization(
        loss_derivative, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs,
        lrWeights, lrBias, num_shots, adadelta_update, epsilon=epsilon, rho=rho
    )
