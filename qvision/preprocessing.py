import cupy as cp

def rgb2gray(rgb):
    """Convert an RGB digital image to grayscale."""
    return cp.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

def convert_to_float(images, labels):
    """Convert images and labels to float64."""
    return images.astype(cp.float64), labels.astype(cp.float64)

def convert_and_normalize(images):
    """Convert images to grayscale and normalize them."""
    for idx in range(images.shape[0]):  # Cambiato per lavorare con gli indici di CuPy
        img = images[idx]
        if len(images.shape) == 4:  # Controllo se le immagini hanno una dimensione di 4 (batch con canale)
            img_gray = rgb2gray(img)
            images[idx, :, :, 0] = img_gray / cp.sum(img_gray)  # Usa cp.sum invece di np.sum
        else:
            img_gray = img
            images[idx, :, :] = img_gray / cp.sum(img_gray)  # Usa cp.sum invece di np.sum
    return images

def calculate_amplitudes(images):
    """Calculate amplitudes of the images."""
    if len(images.shape) == 4:  # Controllo per batch con canale
        return cp.sqrt(images[:, :, :, 0])  # Usa cp.sqrt invece di np.sqrt
    else:
        return cp.sqrt(images[:, :, :])  # Usa cp.sqrt invece di np.sqrt