# Q-Vision/qvision/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns


def plot_loss_accuracy(
        loss_history,
        test_loss_history,
        accuracy_history,
        test_accuracy_history,
        figsize=(18, 8),
        dpi=150,
        font_size=16,
        title_size=20,
        label_size=18,
        legend_size=14,
        tick_size=14,
        linewidth=2.5,
        linestyle_train='-',
        linestyle_val='--',
        grid=True,
        style='whitegrid',
        marker_train=None,  # Impostato a None per default senza marker
        marker_val=None
):
    """
    Plot the training and validation loss and accuracy with enhanced quality and fonts.

    Additional Parameters:
    - marker_train: Marker style for training lines. Default is None.
    - marker_val: Marker style for validation lines. Default is None.
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
    plt.plot(epochs, loss_history, label='Training Loss', linewidth=linewidth, linestyle=linestyle_train,
             marker=marker_train)
    plt.plot(epochs, test_loss_history, label='Validation Loss', linewidth=linewidth, linestyle=linestyle_val,
             marker=marker_val)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training e Validation Loss')
    plt.legend()
    if grid:
        plt.grid(True)

    # Plot dell'accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy_history, label='Training Accuracy', linewidth=linewidth, linestyle=linestyle_train,
             marker=marker_train)
    plt.plot(epochs, test_accuracy_history, label='Validation Accuracy', linewidth=linewidth, linestyle=linestyle_val,
             marker=marker_val)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training e Validation Accuracy')
    plt.legend()
    if grid:
        plt.grid(True)

    plt.tight_layout()
    plt.show()