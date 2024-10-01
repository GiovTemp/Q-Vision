import numpy as np
import collections
from .utils import sig, sigPrime

def amplitude(z):
    return np.abs(z)

def phase(z):
    return np.angle(z)

def gerchberg_saxton(Source, Target, max_iterations=100, tolerance=1e-6):
    """Gerchberg-Saxton algorithm using only real numbers."""
    A = np.fft.ifft2(Target).real  # Use only the real part of the inverse FFT

    for _ in range(max_iterations):
        # Instead of complex exponential, use cosine for real-valued phase approximation
        B = amplitude(Source) * np.cos(phase(A))
        C = np.fft.fft2(B).real  # Keep only real part after FFT
        D = amplitude(Target) * np.cos(phase(C))
        A = np.fft.ifft2(D).real  # Use only the real part of the inverse FFT

        if np.isclose(np.linalg.norm(amplitude(A) - amplitude(Source)), 0, atol=tolerance):
            break

    Retrieved_Phase = np.cos(phase(A))  # Use real-valued phase (cosine of the phase)
    return Retrieved_Phase


def neuron(weights, bias, Img, num_shots, max_iterations=100, tolerance=1e-6):
    """Neuron function using only real number computations."""
    Source = Img
    Target = weights
    Retrieved_Phase = gerchberg_saxton(Source, Target, max_iterations, tolerance)

    # Modulate using real-valued phase (cosine instead of exp(1j * ...))
    modulated_Img = Img * np.cos(Retrieved_Phase)

    # Compute probability using only real parts (no complex conjugates)
    prob = np.sum(modulated_Img * weights)**2  # Use real-valued multiplication

    if num_shots == -1:
        f = prob
    else:
        samples = np.random.choice([0, 1], num_shots, p=[1 - prob, prob])
        counter = collections.Counter(samples)
        f = counter[1] / num_shots

    return sig(f + bias)


def spatial_loss_derivative(output, target, weights, bias, Img):
    """Compute derivative using only real parts of weights and modulated image."""
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
    modulated_Img = Img * np.cos(Retrieved_Phase)  # Use only real part

    # Gradient calculation using real parts only
    g = np.sum(np.multiply(modulated_Img, weights / norm))
    gPrime = (modulated_Img - g * weights / norm) / norm

    fPrime = 2 * g * gPrime  # Only real part

    crossPrime = (F - y) / (F * (1 - F))

    gAbs = np.abs(g)
    weights_derivative = crossPrime * sigPrime(gAbs ** 2 + bias) * fPrime

    bias_derivative = crossPrime * sigPrime(gAbs ** 2 + bias)

    return weights_derivative, bias_derivative


def Fourier_loss_derivative(output, target, weights, bias, Img):
    """Compute the derivative using real parts of Fourier-encoded input."""
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
    modulated_Img = Img * np.cos(Retrieved_Phase)  # Use real part

    g = np.sum(np.multiply(modulated_Img, weights / norm))
    gAbs = np.abs(g)

    gPrime = (modulated_Img - gAbs * weights / norm) / norm
    fPrime = 2 * gAbs * gPrime

    crossPrime = (F - y) / (F * (1 - F))

    weights_derivative = crossPrime * sigPrime(gAbs ** 2 + bias) * fPrime
    bias_derivative = crossPrime * sigPrime(gAbs ** 2 + bias)

    return weights_derivative, bias_derivative
