from keras.datasets import cifar10, mnist
from qvision import QVision
import numpy as np
import matplotlib.pyplot as plt
import os
import optuna
from sklearn.utils import shuffle

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

    # img = np.array(train0sImgs[0, :, :], dtype='float')
    # pixels = img.reshape((28, 28))
    # plt.imshow(pixels, cmap='gray')
    # plt.show()

    trainImgs = np.concatenate((train0sImgs, train1sImgs), axis = 0)
    testImgs = np.concatenate((test0sImgs, test1sImgs), axis = 0)

    # Create the dataset of images and labels (0s and 1s)
    train0Labels = np.zeros(train0sImgs.shape[0])
    train1Labels = np.ones(train1sImgs.shape[0])
    trainLabels = np.concatenate((train0Labels, train1Labels), axis = 0)

    test0Labels = np.zeros(test0sImgs.shape[0])
    test1Labels = np.ones(test1sImgs.shape[0])
    testLabels = np.concatenate((test0Labels, test1Labels), axis = 0)

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
      trainImgs[idx,:,:] = trainImg/np.sum(trainImg) # Normalization
    for idx, testImg in enumerate(testImgs):
      testImgs[idx,:,:] = testImg/np.sum(testImg) # Normalization

    # Amplitudes
    trainImgs = np.sqrt(trainImgs[:, :, :])
    testImgs = np.sqrt(testImgs[:, :, :])

    # Padding from 28x28 to 32x32
    trainImgs = np.pad(trainImgs, ((0,0),(2,2),(2,2)), mode='constant', \
                       constant_values = 0)
    testImgs = np.pad(testImgs, ((0,0),(2,2),(2,2)), mode='constant', \
                       constant_values = 0)

    # img = np.array(trainImgs[0, :, :])
    # pixels = img.reshape((32, 32))
    # plt.imshow(pixels, cmap='gray')
    # plt.show()

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


# Impostazione degli iperparametri
factor = 35
numEpochs = 1200
learningRateWeights = 0.01/factor
learningRateBias = 0.001/factor
numShots = -1

# Inizializzazione della classe QVision
model = QVision(num_epochs=numEpochs, lr_weights=learningRateWeights, lr_bias=learningRateBias, num_shots=numShots)

trainImgs, trainLabels, testImgs, testLabels = load_mnist()

# Inizializzazione dei parametri
model.initialize_parameters()

# Define the folders where the images are stored
train_images_folder = 'training_images'
test_images_folder = 'test_images'

# Load the data
(train_source_images, train_modulated_images, train_labels), \
(test_source_images, test_modulated_images, test_labels) = load_train_test_images(train_images_folder, test_images_folder)

# Shuffle the training set (source images, modulated images, and labels)
train_source_images, train_modulated_images, train_labels = shuffle_dataset(train_source_images, train_modulated_images, train_labels)

# Preprocessamento dei dati (l'utente deve fornire trainImgs, trainLabels, testImgs, testLabels)
# trainImgs, trainLabels, testImgs, testLabels = model.preprocess_data(trainImgs, trainLabels, testImgs, testLabels)

# Training del modello
weights, bias, loss_history, test_loss_history, accuracy_history, test_accuracy_history = model.train('sgd_momentum', None, None, None, None, train_source_images, train_modulated_images, train_labels,
          test_source_images, test_modulated_images, test_labels, phase_modulation=True)

# Visualizzazione dei grafici di perdita e accuratezza
print(loss_history, test_loss_history, accuracy_history, test_accuracy_history)

"""
Plot the training and validation loss and accuracy.

Parameters:
- loss_history: List of training loss values.
- test_loss_history: List of validation loss values.
- accuracy_history: List of training accuracy values.
- test_accuracy_history: List of validation accuracy values.
"""
epochs = range(1, len(loss_history) + 1)

plt.figure(figsize=(14, 6))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(epochs, loss_history, label='Training Loss')
plt.plot(epochs, test_loss_history, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, accuracy_history, label='Training Accuracy')
plt.plot(epochs, test_accuracy_history, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
