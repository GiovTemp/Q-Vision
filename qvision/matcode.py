import numpy as np
import os

def qn2(C, y, lambda_weights, bias, sig_par, B):
    alfa, beta = sig_par
    N = np.size(C)

    lambda_phase = np.exp(2 * np.pi * 1j * lambda_weights) / np.sqrt(N)
    weights = np.fft.fft2(lambda_phase) / np.sqrt(N)
    gg = np.multiply(C, np.conj(weights))
    g = np.sum(gg)
    f = np.abs(g) ** 2
    sig = 1 / (1 + np.exp(-alfa * (f + bias) + beta))

    H = -y * np.log(sig) - (1 - y) * np.log(1 - sig)

    # Derivative calculations
    gPrime = -2 * np.pi * 1j * np.multiply(B, np.conj(lambda_phase)) * N
    fPrime = 2 * np.real(g * np.conj(gPrime))
    sigPrime = sig * (1 - sig)
    HPrime = (sig - y) / (sig * (1 - sig))

    grad_lambda = alfa * HPrime * sigPrime * fPrime
    grad_bias = alfa * HPrime * sigPrime

    return sig, H, grad_lambda, grad_bias

def train_model(epochs, lr, sig_par, train_source_images, train_modulated_images, y, N, lambda_weights, bias):
    loss_history = []
    accuracy_history = []

    for epoch in range(epochs):
        mean_grad_lambda = np.zeros((28, 28))
        mean_grad_bias = 0
        mean_H = 0
        sig = np.zeros(N)

        for i in range(N):
            B = train_source_images[i, :].reshape(28, 28)
            C = train_modulated_images[i, :].reshape(28, 28)
            y_i = y[i]

            sig[i], H, grad_lambda, grad_bias = qn2(C, y_i, lambda_weights, bias, sig_par, B)

            # Update gradient means
            mean_grad_lambda = (1 / (i + 1)) * ((i * mean_grad_lambda) + grad_lambda)
            mean_grad_bias = (1 / (i + 1)) * ((i * mean_grad_bias) + grad_bias)
            mean_H = (1 / (i + 1)) * ((i * mean_H) + H)

        # Update weights and bias using gradient descent
        lambda_weights = lambda_weights - lr[0] * mean_grad_lambda
        bias = bias - lr[1] * mean_grad_bias

        # Store the loss for this epoch
        loss_history.append(mean_H)

        predictions = sig > 0.5
        accuracy = np.sum(predictions == y) / N

        accuracy_history.append(accuracy)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {mean_H:.4f}, Accuracy: {accuracy:.4f}")

    return lambda_weights, bias, loss_history, accuracy_history

def load_images_from_directory(folder_path):
    images = []
    labels = []

    # Iterate over the files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            # Load the image (amplitude values) from the .npy file
            image = np.load(os.path.join(folder_path, filename))

            # Extract the label from the filename
            label = float(filename.split('_')[-1].split('.')[0])

            images.append(image)
            labels.append(label)

    # Convert to NumPy arrays for compatibility with deep learning libraries
    images = np.array(images)
    labels = np.array(labels)

    return images, labels

def load_train_test_images(train_folder, test_folder):
    # Load training source and modulated images
    train_source_folder = f'{train_folder}/source_images'
    train_modulated_folder = f'{train_folder}/modulated_images'

    # Load test source and modulated images
    test_source_folder = f'{test_folder}/source_images'
    test_modulated_folder = f'{test_folder}/modulated_images'

    # Load training images
    train_source_images, train_labels = load_images_from_directory(train_source_folder)
    train_modulated_images, _ = load_images_from_directory(train_modulated_folder)

    # Load testing images
    test_source_images, test_labels = load_images_from_directory(test_source_folder)
    test_modulated_images, _ = load_images_from_directory(test_modulated_folder)

    return (train_source_images, train_modulated_images, train_labels), \
           (test_source_images, test_modulated_images, test_labels)


# Define the folders where the images are stored
train_images_folder = 'training_images'
test_images_folder = 'test_images'

# Parametri di esempio
epochs = 1200
lr = [0.01/35, 0.001/35]
sig_par = [385, 5.3]
N = 2000  # Numero di esempi di allenamento
lambda_weights = np.random.rand(28, 28)  # Inizializzazione casuale di lambda
bias = 0
# train_source_images = np.random.rand(N, 784)  # Immagini sorgente di esempio
# train_modulated_images = np.random.rand(N, 784)  # Immagini modulate di esempio

# Load the data
(train_source_images, train_modulated_images, train_labels), \
(test_source_images, test_modulated_images, test_labels) = load_train_test_images(train_images_folder, test_images_folder)

# Allenamento del modello
lambda_weights, bias, loss_history, accuracy_history = train_model(
    epochs, lr, sig_par, train_source_images, train_modulated_images, train_labels, N, lambda_weights, bias
)
