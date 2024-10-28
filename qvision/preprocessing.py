import cupy as cp

def rgb2gray(rgb):
    """Convert an RGB digital image to grayscale."""
    return cp.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

def convert_to_float(images, labels):
    """Convert images and labels to float64."""
    return images.astype(cp.float64), labels.astype(cp.float64)

def convert_and_normalize(images):
    """Convert images to grayscale and normalize them."""
    # Assicurati che le immagini siano in CuPy
    images = cp.asarray(images)

    for idx in range(images.shape[0]):
        img = images[idx]
        # Controlla se l'immagine è RGB
        if img.ndim == 3 and img.shape[2] == 3:  # Controllo per immagini RGB
            img_gray = rgb2gray(img)
            images[idx, :, :, 0] = img_gray / cp.sum(img_gray) if cp.sum(img_gray) > 0 else img_gray  # Evita divisione per zero
        elif img.ndim == 2:  # Immagine già in scala di grigi
            images[idx, :, :] = img / cp.sum(img) if cp.sum(img) > 0 else img  # Evita divisione per zero
    return images

def calculate_amplitudes(images):
    """Calculate amplitudes of the images."""
    images = cp.asarray(images)  # Assicurati che siano array CuPy
    if len(images.shape) == 4:  # Controllo per batch con canale
        return cp.sqrt(images[:, :, :, 0])  # Usa cp.sqrt
    else:
        return cp.sqrt(images[:, :, :])  # Usa cp.sqrt