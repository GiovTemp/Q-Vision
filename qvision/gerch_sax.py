import numpy as np
import os
from keras.datasets import mnist
import matplotlib.pyplot as plt

def amplitude(z):
    return np.abs(z)

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
        B = I * np.exp(1j * np.angle(A))
        C = np.fft.fft2(B) / np.sqrt(n * m)
        D = F * np.exp(1j * np.angle(C))
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

def load_mnist():
    # Load the MNIST dataset
    (trainImgs, trainLabels), (testImgs, testLabels) = mnist.load_data()

    # Filter 0 and 1 from the dataset
    train0s, test0s = np.where(trainLabels == 0), np.where(testLabels == 0)
    train1s, test1s = np.where(trainLabels == 1), np.where(testLabels == 1)

    train0sImgs = trainImgs[train0s[0]][:1000]
    train1sImgs = trainImgs[train1s[0]][:1000]

    test0sImgs = testImgs[test0s[0]][:100]
    test1sImgs = testImgs[test1s[0]][:100]

    trainImgs = np.concatenate((train0sImgs, train1sImgs), axis=0)
    testImgs = np.concatenate((test0sImgs, test1sImgs), axis=0)

    # Create the dataset of images and labels (0s and 1s)
    train0Labels = np.zeros(train0sImgs.shape[0])
    train1Labels = np.ones(train1sImgs.shape[0])
    trainLabels = np.concatenate((train0Labels, train1Labels), axis=0)

    test0Labels = np.zeros(test0sImgs.shape[0])
    test1Labels = np.ones(test1sImgs.shape[0])
    testLabels = np.concatenate((test0Labels, test1Labels), axis=0)

    # Reshuffle images and labels consistently
    idxs = np.arange(trainImgs.shape[0])
    np.random.shuffle(idxs)

    trainImgs = trainImgs[idxs]
    trainLabels = trainLabels[idxs]

    # Convert to float
    trainImgs = trainImgs.astype(np.float64)
    testImgs = testImgs.astype(np.float64)
    trainLabels = trainLabels.astype(np.float64)
    testLabels = testLabels.astype(np.float64)

    # Identify each image with the single-photon discretized amplitudes
    for idx, trainImg in enumerate(trainImgs):
        trainImgs[idx, :, :] = trainImg / np.sum(trainImg)  # Normalization
    for idx, testImg in enumerate(testImgs):
        testImgs[idx, :, :] = testImg / np.sum(testImg)  # Normalization

    # Amplitudes
    trainImgs = np.sqrt(trainImgs[:, :, :])
    testImgs = np.sqrt(testImgs[:, :, :])

    # # Padding from 28x28 to 32x32
    # trainImgs = np.pad(trainImgs, ((0, 0), (2, 2), (2, 2)), mode='constant', constant_values=0)
    # testImgs = np.pad(testImgs, ((0, 0), (2, 2), (2, 2)), mode='constant', constant_values=0)

    # Reduce the training set
    trainImgs = trainImgs[:, :, :]
    trainLabels = trainLabels[:]

    return trainImgs, trainLabels, testImgs, testLabels

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

if __name__ == '__main__':
    # Load the MNIST dataset
    trainImgs, trainLabels, testImgs, testLabels = load_mnist()

    # Process and save training images
    process_and_save_images(trainImgs, trainLabels, 'training_images')

    # Process and save test images
    process_and_save_images(testImgs, testLabels, 'test_images')
