import numpy as np
import os
from keras.datasets import mnist
import matplotlib.pyplot as plt

def phase(z):
    return np.angle(z)

# Function to plot amplitude and phase of complex images
def plot_complex_image(img, C):
    # Compute amplitude and phase of the complex image
    amplitude_C = np.abs(C)

    # Plot amplitude and phase
    plt.figure(figsize=(12, 6))

    # Amplitude plot
    plt.subplot(1, 2, 1)
    plt.title("Amplitude of Modulated Image (C)")
    plt.imshow(amplitude_C, cmap='gray')
    plt.colorbar()

    # Phase plot
    plt.subplot(1, 2, 2)
    plt.title("Image")
    plt.imshow(img, cmap='gray')
    plt.colorbar()

    plt.show()

def gerch_sax(I, F, N_tot=10):
    n, m = F.shape
    D = F
    trend = np.zeros(N_tot)

    for i in range(N_tot):
        A = np.fft.ifft2(D) * np.sqrt(n * m)
        B = I * np.exp(1j * phase(A))
        C = np.fft.fft2(B) / np.sqrt(n * m)
        D = F * np.exp(1j * phase(C))
        trend[i] = np.linalg.norm(np.abs(C) - np.abs(F))

    return B, C

def save_image_npy(image, path):
    # Save the raw image array as an .npy file to preserve exact values
    np.save(path, image)

def process_and_save_images(images, labels, folder_prefix):
    # Create source and modulated image folders
    source_folder = f'{folder_prefix}/source_images'
    modulated_folder = f'{folder_prefix}/modulated_images'

    os.makedirs(source_folder, exist_ok=True)
    os.makedirs(modulated_folder, exist_ok=True)

    for i, image in enumerate(images):
        label = labels[i]

        # Using the same image as Source and Target for simplicity
        Source = np.ones((image.shape[0], image.shape[1]))
        Source = Source/np.linalg.norm(Source)
        Target = image

        # Apply Gerchberg-Saxton algorithm
        B, C = gerch_sax(Source, Target)

        # plot_complex_image(image, C)
        # print(label)
        #
        # if i == 5:
        #     break

        # Save source image (B) and modulated image (C)
        source_image_path = os.path.join(source_folder, f'image_{i}_label_{int(label)}.png')
        modulated_image_path = os.path.join(modulated_folder, f'image_{i}_label_{int(label)}.png')

        # Save the images as .npy files
        save_image_npy(B, source_image_path)
        save_image_npy(C, modulated_image_path)

        if i % 100 == 0:
            print(f'Images processed: {i}')

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

def shuffle_dataset(source_images, modulated_images, labels):
    """
    Shuffles the source images, modulated images, and labels together, maintaining their correspondence.

    Parameters:
    source_images (numpy array): The original source images.
    modulated_images (numpy array): The modulated images.
    labels (numpy array): The corresponding labels.

    Returns:
    shuffled_source_images, shuffled_modulated_images, shuffled_labels: Shuffled dataset and labels.
    """
    # Concatenate source images, modulated images, and labels into one array to shuffle consistently
    indices = np.arange(len(labels))
    shuffled_indices = np.random.permutation(indices)

    shuffled_source_images = source_images[shuffled_indices]
    shuffled_modulated_images = modulated_images[shuffled_indices]
    shuffled_labels = labels[shuffled_indices]

    return shuffled_source_images, shuffled_modulated_images, shuffled_labels
