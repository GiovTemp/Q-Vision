import collections
import cupy as cp  # Importing CuPy
from typing import Callable

from .utils import sig, sigPrime, loss, accuracy
from .photon_detector import calculate_f_i


class QuantumNeuron:
    def __init__(self):
        self.N = 0  # Inizializza N solo al primo utilizzo

    def neuron(self, weights, bias, Img, num_shots, ideal_conditions, non_ideal_parameters):
        """ Compute the output of the quantum optical neuron, with parameters
            weights and bias, and input Img. The predicted probability is sampled
            for a given number of shots (deactivated by choosing shots = -1). """

        norm = cp.sqrt(cp.sum(cp.square(weights)))  # Use cp
        prob = cp.abs(cp.sum(cp.multiply(Img, weights / norm))) ** 2  # Use cp

        if num_shots == -1:
            f = prob
            f_i, self.N = calculate_f_i(weights, Img, 1, ideal_conditions, non_ideal_parameters, f, self.N)
        else:
            samples = cp.random.choice([0, 1], num_shots, p=[1 - prob, prob])
            counter = collections.Counter(samples.get())
            f = counter[1] / num_shots
            f_i, self.N = calculate_f_i(weights, Img, num_shots, ideal_conditions, non_ideal_parameters, f, self.N)

        return sig(f_i + bias)

    def neuron_phase_modulation(self, weights, bias, modulated_image, num_shots, alfa=385, beta=5.3,
                                max_iterations=100):
        # Normalize modulated image
        modulated_image = modulated_image / cp.linalg.norm(modulated_image)  # Use cp.linalg.norm

        # Compute weights phase
        N = weights.size
        weights_phase = cp.exp(2 * cp.pi * 1j * weights) / cp.sqrt(N)  # Use cp
        weights2 = cp.fft.fft2(weights_phase) / cp.sqrt(N)  # Use cp

        # Compute the overlap (inner product)
        g = cp.sum(cp.multiply(modulated_image, cp.conj(weights2)))  # Use cp
        prob = cp.abs(g) ** 2  # Equivalent to f = abs(g)^2 in MATLAB

        # Calculate sigmoid function with alfa and beta as hyperparameters
        sigmoid_input = -alfa * (prob + bias) + beta
        sig_value = 1 / (1 + cp.exp(sigmoid_input))  # Use cp.exp

        if num_shots == -1:
            f = prob
        else:
            samples = cp.random.choice([0, 1], num_shots, p=[1 - prob, prob])
            counter = collections.Counter(samples)
            f = counter[1] / num_shots

        return sig_value


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
    norm = cp.sqrt(cp.sum(cp.square(weights)))  # Use cp

    # Compute the derivative with respect to the weights
    g = cp.sum(cp.multiply(Img, weights / norm))  # <I, U>
    gPrime = (Img - g * weights / norm) / norm  # <I, dlambdaU>

    fPrime = 2 * cp.real(g * cp.conjugate(gPrime))  # 2Re[<I, U><I, dU>*]

    crossPrime = (F - y) / (F * (1 - F))

    gAbs = cp.abs(g)  # sqrt(f)
    weights_derivative = crossPrime * sigPrime(gAbs ** 2 + bias) * fPrime

    # Compute the derivative with respect to the bias
    bias_derivative = crossPrime * sigPrime(gAbs ** 2 + bias)

    return weights_derivative, bias_derivative


def pm_spatial_loss_derivative(output, target, source_image, modulated_image, weights, bias, alfa=385, beta=5.3):
    """
    Compute the gradient of the binary cross-entropy with respect to the neuron parameters,
    incorporating the phase information.
    """
    # Error checks
    if output == 1:
        raise ValueError('Output is 1!')
    elif output <= 0:
        raise ValueError('Output is negative!')
    elif 1 - output <= 0:
        raise ValueError('Output is greater than 1!')

    F = output  # The neuron's output
    y = target  # The true label

    # Normalize the modulated image
    modulated_image = modulated_image / cp.linalg.norm(modulated_image)  # Use cp.linalg.norm

    # Phase calculations for weights
    N = weights.size
    weights_phase = cp.exp(2 * cp.pi * 1j * weights) / cp.sqrt(N)  # Use cp
    weights2 = cp.fft.fft2(weights_phase) / cp.sqrt(N)  # Use cp

    # Compute g as the inner product between the modulated image and the conjugate of the weights
    g = cp.sum(modulated_image * cp.conj(weights2))  # Use cp
    f = cp.abs(g) ** 2  # f = |g|^2

    # Sigmoid computation
    sig_value = 1 / (1 + cp.exp(-alfa * (f + bias) + beta))  # Use cp.exp

    # Binary cross-entropy loss (H) computation
    H = -y * cp.log(sig_value) - (1 - y) * cp.log(1 - sig_value)  # Use cp.log

    # Calculate gradients
    gPrime = -2 * cp.pi * 1j * source_image * cp.conj(weights_phase) * N  # Use cp
    fPrime = 2 * cp.real(g * cp.conjugate(gPrime))  # Use cp
    sigPrime_value = sig_value * (1 - sig_value)
    HPrime = (sig_value - y) / (sig_value * (1 - sig_value))

    grad_lambda = alfa * HPrime * sigPrime_value * fPrime
    grad_bias = alfa * HPrime * sigPrime_value

    return grad_lambda, grad_bias


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
    norm = cp.sqrt(cp.sum(cp.square(weights)))  # Use cp

    # Compute the derivative with respect to the weights
    g = cp.sum(cp.multiply(Img, weights / norm))  # <I, U>
    gAbs = cp.abs(g)  # sqrt(f)

    gPrime = (Img - gAbs * weights / norm) / norm  # Approximation
    fPrime = 2 * cp.real(gAbs * cp.conjugate(gPrime))  # Approximation

    crossPrime = (F - y) / (F * (1 - F))

    weights_derivative = crossPrime * sigPrime(gAbs ** 2 + bias) * fPrime

    # Compute the derivative with respect to the bias
    bias_derivative = crossPrime * sigPrime(gAbs ** 2 + bias)

    return weights_derivative, bias_derivative


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
        loss_derivative: Callable, weights, bias, targets, test_targets, trainImgs, testImgs, num_epochs,
        train_source_images,
        train_modulated_images, train_labels,
        test_source_images, test_modulated_images, test_labels, lrWeights, lrBias, num_shots,
        update_fn: Callable, ideal_conditions, non_ideal_parameters, phase_modulation, **kwargs
):
    """Common optimization loop with handling for complex numbers due to phase modulation and multiple optimizers."""

    # History initialization

    loss_history = []
    accuracy_history = []

    test_loss_history = []
    test_accuracy_history = []

    # Cache initialization for momentum or other optimizers that need cache
    cache = kwargs.get('cache', {})
    momentum = kwargs.get('momentum', 0.9)

    # Inizializza la classe QuantumNeuron una sola volta
    neuron_model = QuantumNeuron()

    if phase_modulation:
        # Retrieve batch size from kwargs, default to full-batch if not specified
        training_images_length = train_modulated_images.shape[0]
        test_images_length = test_modulated_images.shape[0]
        batch_size = kwargs.get('batch_size', training_images_length)
        trainlabels = train_labels
        testlabels = test_labels
    else:
        # Retrieve batch size from kwargs, default to full-batch if not specified
        training_images_length = trainImgs.shape[0]
        test_images_length = testImgs.shape[0]
        batch_size = kwargs.get('batch_size', training_images_length)
        trainlabels = targets
        testlabels = test_targets

    for epoch in range(num_epochs):
        # Shuffle if using mini-batch or SGD, keep order for standard GD
        indices = cp.arange(training_images_length)  # Use cp
        if batch_size < training_images_length:  # Only shuffle for mini-batch or SGD
            cp.random.shuffle(indices)

        # Divide into batches
        for start_idx in range(0, training_images_length, batch_size):
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]

            batch_lossWeightDerivatives = []
            batch_lossBiasDerivatives = []

            # Calculate gradients for each sample in batch
            for idx in batch_indices:
                if phase_modulation:
                    output = neuron_model.neuron_phase_modulation(weights, bias, train_modulated_images[idx, :, :],
                                                                  num_shots)

                    # Compute loss derivatives with respect to weights and bias
                    lossWeightDerivative, lossBiasDerivative = pm_spatial_loss_derivative(
                        cp.abs(output), trainlabels[idx], train_source_images[idx, :, :],
                        train_modulated_images[idx, :, :], weights, bias
                    )
                else:
                    output = neuron_model.neuron(weights, bias, trainImgs[idx, :, :], num_shots, ideal_conditions,
                                                 non_ideal_parameters)

                    # Compute loss derivatives with respect to weights and bias
                    lossWeightDerivative, lossBiasDerivative = loss_derivative(
                        cp.abs(output), trainlabels[idx], weights, bias, trainImgs[idx, :, :]
                    )

                batch_lossWeightDerivatives.append(lossWeightDerivative)
                batch_lossBiasDerivatives.append(lossBiasDerivative)

                if update_fn == sgd_update:
                    weights, bias, cache = update_fn(
                        weights, bias, lossWeightDerivative, lossBiasDerivative, lrWeights, lrBias, cache
                    )
                elif update_fn == sgd_momentum_update:
                    weights, bias, cache = update_fn(
                        weights, bias, lossWeightDerivative, lossBiasDerivative, lrWeights, lrBias, cache, momentum
                    )
                elif batch_size > 1 and batch_size < training_images_length and update_fn == standard_gd_update:
                    weights, bias, cache = update_fn(
                        weights, bias, batch_lossWeightDerivatives, batch_lossBiasDerivatives, lrWeights, lrBias, cache
                    )

            # Update weights and biases
            if update_fn == standard_gd_update and batch_size == training_images_length:
                weights, bias, cache = update_fn(
                    weights, bias, batch_lossWeightDerivatives, batch_lossBiasDerivatives, lrWeights, lrBias, cache
                )

        # Calculate losses and accuracy after each epoch
        if phase_modulation:
            outputs = cp.array([
                neuron_model.neuron_phase_modulation(weights, bias, train_modulated_images[idx, :, :], num_shots)
                for idx in range(training_images_length)
            ], dtype=cp.complex128)  # Use cp
        else:
            outputs = cp.array([
                neuron_model.neuron(weights, bias, trainImgs[idx, :, :], num_shots, ideal_conditions,
                                    non_ideal_parameters)
                for idx in range(training_images_length)
            ])

        # Calculate training loss and accuracy
        losses = cp.array([loss(cp.abs(outputs[idx]), trainlabels[idx]) for idx in range(outputs.shape[0])])  # Use cp
        loss_history.append(cp.mean(losses))  # Use cp
        accuracy_history.append(accuracy(outputs, trainlabels))

        # Calculate test losses and accuracy
        if phase_modulation:
            test_outputs = cp.array([
                neuron_model.neuron_phase_modulation(weights, bias, test_modulated_images[idx, :, :], num_shots)
                for idx in range(test_images_length)
            ], dtype=cp.complex128)  # Use cp
        else:
            test_outputs = cp.array([
                neuron_model.neuron(weights, bias, testImgs[idx, :, :], num_shots, ideal_conditions,
                                    non_ideal_parameters)
                for idx in range(test_images_length)
            ])

        test_losses = cp.array(
            [loss(cp.abs(test_outputs[idx]), testlabels[idx]) for idx in range(test_outputs.shape[0])])  # Use cp
        test_loss_history.append(cp.mean(test_losses))  # Use cp
        test_accuracy_history.append(accuracy(test_outputs, testlabels))

        print('EPOCH', epoch + 1)
        print('Loss', loss_history[-1], 'Val_Loss', test_loss_history[-1])
        print('Accuracy', accuracy_history[-1], 'Val_Acc', test_accuracy_history[-1])
        print('---')

    return weights, bias, loss_history, test_loss_history, accuracy_history, test_accuracy_history


# Define the standard gradient descent update function
def standard_gd_update(weights, bias, lossWeightsDerivatives, lossBiasDerivatives, lrWeights, lrBias, cache):
    """ Parameters update rule of the gradient descent algorithm. """
    clipped_gradients_weights = clip_gradients(lossWeightsDerivatives)
    clipped_gradients_bias = clip_gradients(lossBiasDerivatives)
    new_weights = weights - lrWeights * cp.mean(clipped_gradients_weights, axis=0)  # Use cp
    new_bias = bias - lrBias * cp.mean(clipped_gradients_bias, axis=0)  # Use cp
    return new_weights, new_bias, cache


# Define the SGD update function
def sgd_update(weights, bias, lossWeightsDerivatives, lossBiasDerivatives, lrWeights, lrBias, cache):
    clipped_gradients_weights = clip_gradients(lossWeightsDerivatives)
    clipped_gradients_bias = clip_gradients(lossBiasDerivatives)
    weights -= lrWeights * clipped_gradients_weights
    bias -= lrBias * clipped_gradients_bias
    return weights, bias, cache


def sgd_momentum_update(weights, bias, lossWeightsDerivatives, lossBiasDerivatives, lrWeights, lrBias, cache,
                        momentum=0.9):
    """Update rule for Stochastic Gradient Descent (SGD) with Momentum."""

    # Ensure derivatives are treated as at least 1D arrays
    lossWeightsDerivatives = cp.atleast_1d(lossWeightsDerivatives)  # Use cp
    lossBiasDerivatives = cp.atleast_1d(lossBiasDerivatives)  # Use cp

    # Retrieve previous velocities from the cache, or initialize if not present
    velocity_weights = cache.get('velocity_weights', cp.zeros_like(weights))  # Use cp
    velocity_bias = cache.get('velocity_bias', cp.zeros_like(bias))  # Use cp

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
    gradients = cp.clip(gradients, -clip_value, clip_value)  # Use cp
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