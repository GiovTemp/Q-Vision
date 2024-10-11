import numpy as np

def amplitude(z):
    return np.abs(z)

def phase(z):
    return np.angle(z)

def gerchberg_saxton(Source, Target, max_iterations=1000, tolerance=5.2e-17):

    target_size = np.sqrt(Target.shape[0] * Target.shape[1])
    A = np.fft.ifft2(Target) * target_size

    for i in range(max_iterations):
        B = amplitude(Source) * np.exp(1j * phase(A))
        C = np.fft.fft2(B) / target_size
        D = amplitude(Target) * np.exp(1j * phase(C))
        A = np.fft.ifft2(D) * target_size
        #A = amplitude(Source) * np.exp(1j * phase(A))

        # Check for convergence
        error = np.linalg.norm(amplitude(C) - amplitude(Target))

        if i % 50 == 0:
            print(error)

        if error < tolerance:
            break

    Retrieved_Phase = phase(A)
    return B, C

source = np.ones((5, 5))
source = source/np.linalg.norm(source)

target = np.random.rand(5, 5)
target = target/np.linalg.norm(target)

B, C = gerchberg_saxton(source, target)

print('B:', B)
print('C:', C)


