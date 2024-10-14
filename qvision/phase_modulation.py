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

        # print(f'{i}: {error}')

        if error < tolerance:
            break

    Retrieved_Phase = phase(A)
    return B, C

def neuron(weights, bias, Img, modulated_image, num_shots, max_iterations=100):

    # Source = np.ones((Img.shape[0], Img.shape[1]))
    # Source = Source/np.linalg.norm(Source)
    # Target = np.sqrt(Img)
    # Target = Target / np.linalg.norm(Target)

    # _, modulated_image = gerchberg_saxton(Source, Target, max_iterations)
    modulated_image = modulated_image / np.linalg.norm(modulated_image)

    weights_phase = np.exp(2 * math.pi * 1j * weights)
    weights_phase = weights_phase / np.linalg.norm(weights_phase)

    weights2 = np.fft.fft2(weights_phase)
    weights2 = weights2 / np.linalg.norm(weights2)
    prob = np.abs(np.sum(np.multiply(modulated_image, np.conj(weights2))))**2

    #print('prob:', prob)

    # print('norma di modulated_image:', np.linalg.norm(modulated_image))
    # print('norma di weights2:', np.linalg.norm(weights2))
    #
    # print('prob:', prob)

    if num_shots == -1:
        f = prob
    else:
        samples = np.random.choice([0, 1], num_shots, p=[1 - prob, prob])
        counter = collections.Counter(samples)
        f = counter[1]/num_shots
    return sig(f + bias)

def spatial_loss_derivative(output, target, source_image, modulated_image, weights, bias, Img):
    if output == 1:
        raise ValueError('Output is 1!')
    elif output <= 0:
        raise ValueError('Output is negative!')
    elif 1 - output <= 0:
        raise ValueError('Output is greater than 1!')

    F = output
    y = target
    norm = np.sqrt(np.sum(np.square(weights)))

    img_size = np.sqrt(Img.shape[0] * Img.shape[1])

    # Source = np.ones((Img.shape[0], Img.shape[1]))
    # Source = Source/np.linalg.norm(Source)
    # Target = np.sqrt(Img)
    # Target = Target / np.linalg.norm(Target)

    # source_image, modulated_image = gerchberg_saxton(Source, Target, 1000)
    modulated_image = modulated_image / np.linalg.norm(modulated_image)

    weights_phase = np.exp(2 * math.pi * 1j * weights)
    weights_phase = weights_phase / np.linalg.norm(weights_phase)

    weights2 = np.fft.fft2(np.exp(2 * math.pi * 1j * weights))
    weights2 = weights2 / np.linalg.norm(weights2)

    g = np.sum(np.multiply(modulated_image, np.conj(weights2)))  # <I, U>
    gPrime = -2 * math.pi * 1j * np.multiply(source_image, np.conj(weights_phase)) * img_size**2 # <I, dlambdaU>

    fPrime = 2 * np.real(g * np.conjugate(gPrime))  # 2Re[<I, U><I, dU>*]

    crossPrime = (F - y) / (F * (1 - F))

    gAbs = np.abs(g)  # sqrt(f)
    weights_derivative = crossPrime * sigPrime(gAbs ** 2 + bias) * fPrime

    bias_derivative = crossPrime * sigPrime(gAbs ** 2 + bias)

    return weights_derivative, bias_derivative

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
