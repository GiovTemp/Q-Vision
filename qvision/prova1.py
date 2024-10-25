import numpy as np
import os
import matplotlib.pyplot as plt

def plot_complex_image(C):
    # Compute amplitude and phase of the complex image
    amplitude_C = np.abs(C)

    # Plot amplitude and phase
    plt.figure(figsize=(12, 6))

    # Amplitude plot
    plt.subplot(1, 2, 1)
    plt.title("Amplitude of Modulated Image (C)")
    plt.imshow(amplitude_C, cmap='gray')
    plt.colorbar()

    plt.show()

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

    # # Load testing images
    # test_source_images, test_labels = load_images_from_directory(test_source_folder)
    # test_modulated_images, _ = load_images_from_directory(test_modulated_folder)

    return train_source_images, train_modulated_images, train_labels


# Define the folders where the images are stored
train_images_folder = 'training_images'
test_images_folder = 'test_images'

# Load the data
train_source_images, train_modulated_images, train_labels = load_train_test_images(train_images_folder, test_images_folder)

for i in range(10):
    plot_complex_image(train_modulated_images[i, :, :])
    print(train_labels[i])
