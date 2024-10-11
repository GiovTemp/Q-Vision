from keras.datasets import cifar10, mnist
from qvision import QVision
import numpy as np
import matplotlib.pyplot as plt

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
    trainImgs = np.sqrt(trainImgs[:,:,:])
    testImgs = np.sqrt(testImgs[:,:,:])

    # Padding from 28x28 to 32x32
    trainImgs = np.pad(trainImgs, ((0,0),(2,2),(2,2)), mode='constant', \
                       constant_values = 0)
    testImgs = np.pad(testImgs, ((0,0),(2,2),(2,2)), mode='constant', \
                       constant_values = 0)

    # Reduce the training set
    trainImgs = trainImgs[:5000, :, :]
    trainLabels = trainLabels[:5000]

    return trainImgs, trainLabels, testImgs, testLabels

# Impostazione degli iperparametri
numEpochs = 100
learningRateWeights = 0.075
learningRateBias = 0.005
numShots = -1
# learningRateWeights = 0.022811257645838745
# learningRateBias = 0.007159442638014556
# numShots = -1

# Inizializzazione della classe QVision
model = QVision(num_epochs=numEpochs, lr_weights=learningRateWeights, lr_bias=learningRateBias, num_shots=numShots)

trainImgs, trainLabels, testImgs, testLabels = load_mnist()

# Inizializzazione dei parametri
model.initialize_parameters()

# Preprocessamento dei dati (l'utente deve fornire trainImgs, trainLabels, testImgs, testLabels)
#trainImgs, trainLabels, testImgs, testLabels = model.preprocess_data(trainImgs, trainLabels, testImgs, testLabels)

# Training del modello
weights, bias, loss_history, test_loss_history, accuracy_history, test_accuracy_history = model.train('gd', trainImgs, trainLabels, testImgs, testLabels, phase_modulation=True)

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
