from keras.datasets import cifar10
from qvision import QVision
import numpy as np
import matplotlib.pyplot as plt

def load_cifar():
    # Load the CIFAR dataset
    (trainImgs, trainLabels), (testImgs, testLabels) = cifar10.load_data()

    # Filter dogs and planes from the the dataset
    trainCats, testCats = np.where(trainLabels == 3), np.where(testLabels == 3)
    trainDogs, testDogs = np.where(trainLabels == 5), np.where(testLabels == 5)

    trainCatImgs = trainImgs[trainCats[0]]
    trainDogImgs = trainImgs[trainDogs[0]]

    testCatImgs = testImgs[testCats[0]]
    testDogImgs = testImgs[testDogs[0]]

    trainImgs = np.concatenate((trainCatImgs, trainDogImgs), axis = 0)
    testImgs = np.concatenate((testCatImgs, testDogImgs), axis = 0)

    # Assign 0 to cats and 1 to dogs
    trainCatLabels = np.zeros(trainCatImgs.shape[0])
    trainDogLabels = np.ones(trainDogImgs.shape[0])
    trainLabels = np.concatenate((trainCatLabels, trainDogLabels), axis = 0)

    testCatLabels = np.zeros(testCatImgs.shape[0])
    testDogLabels = np.ones(testDogImgs.shape[0])
    testLabels = np.concatenate((testCatLabels, testDogLabels), axis = 0)

    # Reshuffle images and labels consistently
    idxs = np.arange(trainImgs.shape[0])
    np.random.shuffle(idxs)

    trainImgs = trainImgs[idxs]
    trainLabels = trainLabels[idxs]

    # Reduce the training set
    trainImgs = trainImgs[:, :, :]
    trainLabels = trainLabels[:]

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

trainImgs, trainLabels, testImgs, testLabels = load_cifar()

# Inizializzazione dei parametri
model.initialize_parameters()

# Preprocessamento dei dati (l'utente deve fornire trainImgs, trainLabels, testImgs, testLabels)
trainImgs, trainLabels, testImgs, testLabels = model.preprocess_data(trainImgs, trainLabels, testImgs, testLabels)

# Training del modello
weights, bias, loss_history, test_loss_history, accuracy_history, test_accuracy_history = model.train('sgd', trainImgs, trainLabels, testImgs, testLabels, phase_modulation=True)

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
