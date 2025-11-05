"""Median filter denoiser.

Best for impulse noise (salt-and-pepper). Preserves edges well.
Returns uint8 image in [0,255] range.
"""
import numpy as np
from scipy.ndimage import median_filter


def denoise(image: np.ndarray, size: int = 3) -> np.ndarray:
    """Apply median filter to remove impulse noise.

    Parameters
    ----------
    image: np.ndarray
        Grayscale image (uint8 [0,255], float [0,1], or arbitrary range)
    size: int
        Neighborhood size (odd integer recommended, e.g., 3, 5, 7)

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

    out = median_filter(img, size=size)

    if should_rescale:
        out = np.clip(out, 0.0, 1.0)
        return (out * 255).astype(np.uint8)
    else:
        # Return in original arbitrary range
        return out

    # Convert to uint8 [0,255]
    return (out * 255).astype(np.uint8)
