import numpy as np
import os
from keras.datasets import mnist

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
        B, C = gerchberg_saxton(Source, Target)

        # Save source image (B) and modulated image (C)
        source_image_path = os.path.join(source_folder, f'source_image_{i}_label_{int(label)}.png')
        modulated_image_path = os.path.join(modulated_folder, f'modulated_image_{i}_label_{int(label)}.png')

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

    train0sImgs = trainImgs[train0s[0]]
    train1sImgs = trainImgs[train1s[0]]

    test0sImgs = testImgs[test0s[0]]
    test1sImgs = testImgs[test1s[0]]

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

    # Padding from 28x28 to 32x32
    trainImgs = np.pad(trainImgs, ((0, 0), (2, 2), (2, 2)), mode='constant', constant_values=0)
    testImgs = np.pad(testImgs, ((0, 0), (2, 2), (2, 2)), mode='constant', constant_values=0)

    # Reduce the training set
    trainImgs = trainImgs[:, :, :]
    trainLabels = trainLabels[:]

    return trainImgs, trainLabels, testImgs, testLabels

if __name__ == '__main__':
    # Load the MNIST dataset
    trainImgs, trainLabels, testImgs, testLabels = load_mnist()

    # Process and save training images
    process_and_save_images(trainImgs, trainLabels, 'training_images')

    # Process and save test images
    process_and_save_images(testImgs, testLabels, 'test_images')
