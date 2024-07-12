

import numpy as np

def rgb2gray(rgb):
    """Convert an RGB digital image to grayscale."""
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def convert_to_float(images, labels):
    """Convert images and labels to float64."""
    return images.astype(np.float64), labels.astype(np.float64)

def convert_and_normalize(images):
    """Convert images to grayscale and normalize them."""
    for idx, img in enumerate(images):
        img_gray = rgb2gray(img)
        images[idx, :, :, 0] = img_gray / np.sum(img_gray)
    return images

def calculate_amplitudes(images):
    """Calculate amplitudes of the images."""
    return np.sqrt(images[:, :, :, 0])