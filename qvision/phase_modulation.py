import numpy as np
import collections
from .utils import sig, sigPrime
import matplotlib.pyplot as plt

def amplitude(z):
    return np.abs(z)

def phase(z):
    return np.angle(z)

def gerchberg_saxton(Source, Target, max_iterations=100, tolerance=5.2e-17):
    A = np.fft.ifft2(Target)

    for i in range(max_iterations):
        B = amplitude(Source) * np.exp(1j * phase(A))
        C = np.fft.fft2(B)
        D = amplitude(Target) * np.exp(1j * phase(C))
        A = np.fft.ifft2(D)
        A = amplitude(Source) * np.exp(1j * phase(A))

        # Check for convergence
        error = np.linalg.norm(amplitude(A) - amplitude(Source))
        #print(f"Iteration {i}, Error: {error}")

        if error < tolerance:
            #print(f"Converged in {i} iterations with error {error}")
            break

    Retrieved_Phase = phase(A)
    return Retrieved_Phase

def neuron(weights, bias, Img, num_shots, max_iterations=100):
    Source = Img
    Target = weights
    Retrieved_Phase = gerchberg_saxton(Source, Target, max_iterations)
    modulated_Img = Img * np.exp(1j * Retrieved_Phase)
    prob = np.abs(np.sum(modulated_Img * np.conj(weights)))**2

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
    # retrieved_weights = np.abs(modulated_Img)

    g = np.sum(np.multiply(modulated_Img, weights / norm))  # <I, U>
    gPrime = (modulated_Img - g * weights / norm) / norm  # <I, dlambdaU>

    fPrime = 2 * np.real(g * np.conjugate(gPrime))  # 2Re[<I, U><I, dU>*]

    crossPrime = (F - y) / (F * (1 - F))

    gAbs = np.abs(g)  # sqrt(f)
    weights_derivative = crossPrime * sigPrime(gAbs ** 2 + bias) * fPrime

    bias_derivative = crossPrime * sigPrime(gAbs ** 2 + bias)

    return weights_derivative, bias_derivative

def Fourier_loss_derivative(output, target, weights, bias, Img):
    """ Compute the derivative of the binary cross-entropy with respect to the
        neuron parameters, with Fourier-encoded input. """
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
