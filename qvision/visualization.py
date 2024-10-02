# Q-Vision/qvision/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns


def plot_loss_accuracy(
        loss_history,
        test_loss_history,
        accuracy_history,
        test_accuracy_history,
        figsize=(16, 7),
        dpi=100,
        font_size=14,
        title_size=16,
        label_size=14,
        legend_size=12,
        tick_size=12,
        linewidth=2,
        linestyle_train='-',
        linestyle_val='--',
        grid=True,
        style='whitegrid',
        color_train='blue',
        color_val='orange',
        marker_train='o',
        marker_val='s'
):
    """
    Plot the training and validation loss and accuracy with customizable quality and fonts.

    Additional Parameters:
    - color_train: Color for training lines. Default is 'blue'.
    - color_val: Color for validation lines. Default is 'orange'.
    - marker_train: Marker style for training lines. Default is 'o'.
    - marker_val: Marker style for validation lines. Default is 's'.
    """
    # Imposta lo stile del grafico
    sns.set(style=style)

    # Configura le dimensioni dei font
    plt.rcParams.update({
        'font.size': font_size,
        'axes.titlesize': title_size,
        'axes.labelsize': label_size,
        'legend.fontsize': legend_size,
        'xtick.labelsize': tick_size,
        'ytick.labelsize': tick_size
    })

    epochs = range(1, len(loss_history) + 1)

    # Crea una figura con alta risoluzione
    plt.figure(figsize=figsize, dpi=dpi)

    # Plot della loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_history, label='Training Loss',
             linewidth=linewidth, linestyle=linestyle_train,
             color=color_train, marker=marker_train)
    plt.plot(epochs, test_loss_history, label='Validation Loss',
             linewidth=linewidth, linestyle=linestyle_val,
             color=color_val, marker=marker_val)
    plt.xlabel('Epoche')
    plt.ylabel('Loss')
    plt.title('Training e Validation Loss')
    plt.legend()
    if grid:
        plt.grid(True)

    # Plot dell'accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy_history, label='Training Accuracy',
             linewidth=linewidth, linestyle=linestyle_train,
             color=color_train, marker=marker_train)
    plt.plot(epochs, test_accuracy_history, label='Validation Accuracy',
             linewidth=linewidth, linestyle=linestyle_val,
             color=color_val, marker=marker_val)
    plt.xlabel('Epoche')
    plt.ylabel('Accuracy')
    plt.title('Training e Validation Accuracy')
    plt.legend()
    if grid:
        plt.grid(True)

    plt.tight_layout()
    plt.show()