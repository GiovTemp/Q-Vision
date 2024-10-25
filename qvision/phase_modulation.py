import numpy as np
import collections
from .utils import sig, sigPrime
import math
import matplotlib.pyplot as plt

def amplitude(z):
    return np.abs(z)

def phase(z):
    return np.angle(z)

def gerchberg_saxton(Source, Target, max_iterations=1000, tolerance=0.35):

    target_size = np.sqrt(Target.shape[0] * Target.shape[1])
    A = np.fft.ifft2(Target) * target_size

    for i in range(max_iterations):
        B = amplitude(Source) * np.exp(1j * phase(A))
        C = np.fft.fft2(B) / target_size
        D = amplitude(Target) * np.exp(1j * phase(C))
        A = np.fft.ifft2(D) * target_size

        # Check for convergence
        error = np.linalg.norm(amplitude(C) - amplitude(Target))

        if error < tolerance:
            break

    Retrieved_Phase = phase(A)
    return B, C

def neuron(weights, bias, Img, modulated_image, num_shots, alfa=385, beta=5.3, max_iterations=100):
    # Normalize modulated image
    modulated_image = modulated_image / np.linalg.norm(modulated_image)

    # Compute weights phase
    N = weights.size
    weights_phase = np.exp(2 * np.pi * 1j * weights) / np.sqrt(N)
    weights2 = np.fft.fft2(weights_phase) / np.sqrt(N)

    # Compute the overlap (inner product)
    g = np.sum(np.multiply(modulated_image, np.conj(weights2)))
    prob = np.abs(g) ** 2  # Equivalent to f = abs(g)^2 in MATLAB

    # Calculate sigmoid function with alfa and beta as hyperparameters
    sigmoid_input = -alfa * (prob + bias) + beta
    sig_value = 1 / (1 + np.exp(sigmoid_input))

    if num_shots == -1:
        f = prob
    else:
        samples = np.random.choice([0, 1], num_shots, p=[1 - prob, prob])
        counter = collections.Counter(samples)
        f = counter[1] / num_shots

    return sig_value

def spatial_loss_derivative(output, target, source_image, modulated_image, weights, bias, Img, alfa=385, beta=5.3):
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
    modulated_image = modulated_image / np.linalg.norm(modulated_image)

    # Phase calculations for weights
    N = weights.size
    weights_phase = np.exp(2 * np.pi * 1j * weights) / np.sqrt(N)
    weights2 = np.fft.fft2(weights_phase) / np.sqrt(N)

    # Compute g as the inner product between the modulated image and the conjugate of the weights
    g = np.sum(modulated_image * np.conj(weights2))
    f = np.abs(g) ** 2  # f = |g|^2

    # Sigmoid computation
    sig_value = 1 / (1 + np.exp(-alfa * (f + bias) + beta))

    # Binary cross-entropy loss (H) computation
    H = -y * np.log(sig_value) - (1 - y) * np.log(1 - sig_value)

    # Calculate gradients
    gPrime = -2 * np.pi * 1j * source_image * np.conj(weights_phase) * N
    fPrime = 2 * np.real(g * np.conjugate(gPrime))
    sigPrime_value = sig_value * (1 - sig_value)
    HPrime = (sig_value - y) / (sig_value * (1 - sig_value))

    grad_lambda = alfa * HPrime * sigPrime_value * fPrime
    grad_bias = alfa * HPrime * sigPrime_value

    return grad_lambda, grad_bias

def Fourier_loss_derivative(output, target, weights, bias, Img):
    if output == 1:
        raise ValueError('Output is 1!')
    elif output <= 0:
        raise ValueError('Output is negative!')
    elif 1 - output <= 0:
        raise ValueError('Output is greater than 1!')

    F = output
    y = target
    norm = np.sqrt(np.sum(np.square(weights)))

    Source = Img
    Target = weights
    Retrieved_Phase = gerchberg_saxton(Source, Target)
    modulated_Img = Img * np.exp(1j * Retrieved_Phase)

    g = np.sum(np.multiply(modulated_Img, weights / norm))  # <I, U>
    gAbs = np.abs(g)  # sqrt(f)

    gPrime = (modulated_Img - gAbs * weights / norm) / norm  # Approximation
    fPrime = 2 * np.real(gAbs * np.conjugate(gPrime))  # Approximation

    crossPrime = (F - y) / (F * (1 - F))

    weights_derivative = crossPrime * sigPrime(gAbs ** 2 + bias) * fPrime

    bias_derivative = crossPrime * sigPrime(gAbs ** 2 + bias)

    return weights_derivative, bias_derivative
