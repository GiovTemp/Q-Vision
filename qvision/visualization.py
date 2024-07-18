# Q-Vision/qvision/visualization.py

import matplotlib.pyplot as plt


def plot_loss_accuracy(loss_history, test_loss_history, accuracy_history, test_accuracy_history):
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