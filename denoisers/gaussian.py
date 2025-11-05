"""Gaussian spatial denoiser.

Applies Gaussian blur to smooth noise. Best for AWGN (Additive White Gaussian Noise).
Returns uint8 image in [0,255] range.
"""
from scipy.ndimage import gaussian_filter
import numpy as np


def denoise(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian blur denoising.

    Parameters
    ----------
    image: np.ndarray
        Grayscale image (uint8 [0,255], float [0,1], or arbitrary range)
    sigma: float
        Standard deviation for Gaussian kernel

    Returns
    -------
    np.ndarray
        Denoised image (uint8 [0,255] or float for arbitrary range)
    """
    # Handle different input ranges
    if image.dtype == np.uint8:
        img = image.astype(np.float32) / 255.0
        should_rescale = True
    elif image.max() > 1.0 or image.min() < 0.0:
        # Arbitrary range (e.g., log domain)
        img = image.astype(np.float32)
        should_rescale = False
    else:
        img = image.astype(np.float32)
        should_rescale = True

    out = gaussian_filter(img, sigma=sigma)

    if should_rescale:
        out = np.clip(out, 0.0, 1.0)
        return (out * 255).astype(np.uint8)
    else:
        # Return in original arbitrary range
        return out
