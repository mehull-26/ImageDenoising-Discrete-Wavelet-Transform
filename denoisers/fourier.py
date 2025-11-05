"""Fourier-domain low-pass denoiser.

Removes high-frequency components (assumed to be noise) in frequency domain.
Best for noise that appears as high-frequency artifacts.
Returns uint8 image in [0,255] range.

Provides several low-pass filters:
- ideal: circular hard cutoff
- gaussian: Gaussian low-pass (soft, recommended)
- raised_cosine: smooth roll-off using cosine taper
"""
import numpy as np


def _ideal_mask(rows, cols, cutoff_fraction):
    crow, ccol = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    dist = np.sqrt((Y - crow)**2 + (X - ccol)**2)
    max_radius = np.sqrt(crow**2 + ccol**2)
    cutoff = cutoff_fraction * max_radius
    return dist <= cutoff


def _gaussian_mask(rows, cols, cutoff_fraction, sigma_fraction=None):
    # Gaussian low-pass centered at DC. cutoff_fraction controls the nominal radius
    crow, ccol = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    dist = np.sqrt((Y - crow)**2 + (X - ccol)**2)
    max_radius = np.sqrt(crow**2 + ccol**2)
    cutoff = cutoff_fraction * max_radius
    # sigma controls spread; default: cutoff/2
    sigma = (cutoff / 2.0) if sigma_fraction is None else (sigma_fraction * max_radius)
    # Gaussian: exp(-dist^2/(2*sigma^2)) but normalized by max_radius
    return np.exp(-(dist ** 2) / (2.0 * (sigma ** 2)))


def _raised_cosine_mask(rows, cols, cutoff_fraction, rolloff_fraction=0.05):
    crow, ccol = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    dist = np.sqrt((Y - crow)**2 + (X - ccol)**2)
    max_radius = np.sqrt(crow**2 + ccol**2)
    cutoff = cutoff_fraction * max_radius
    # rolloff: fractional width of transition band (as fraction of max_radius)
    rolloff = rolloff_fraction * max_radius
    mask = np.zeros_like(dist, dtype=float)
    inner = dist <= (cutoff - rolloff)
    outer = dist >= (cutoff + rolloff)
    trans = (~inner) & (~outer)
    mask[inner] = 1.0
    # cosine taper in transition band
    mask[trans] = 0.5 * (1.0 + np.cos(np.pi *
                         (dist[trans] - (cutoff - rolloff)) / (2 * rolloff)))
    return mask


def denoise(image: np.ndarray, cutoff_fraction: float = 0.1, filter_type: str = "gaussian",
            sigma_fraction: float = None, rolloff_fraction: float = 0.05) -> np.ndarray:
    """Apply a frequency-domain low-pass filter.

    Parameters
    ----------
    image: np.ndarray
        Grayscale image (uint8 [0,255] or float [0,1])
    cutoff_fraction: float
        Fraction of max radius for cutoff (0.05-0.3 typical)
    filter_type: str
        'ideal', 'gaussian' (recommended), or 'raised_cosine'
    sigma_fraction: float or None
        For gaussian filter, fraction of max_radius as sigma
    rolloff_fraction: float
        For raised_cosine, transition band width

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

    rows, cols = img.shape
    F = np.fft.fftshift(np.fft.fft2(img))

    ft = filter_type.lower()
    if ft == "ideal":
        mask = _ideal_mask(rows, cols, cutoff_fraction).astype(float)
    elif ft == "gaussian":
        mask = _gaussian_mask(rows, cols, cutoff_fraction, sigma_fraction)
    elif ft == "raised_cosine":
        mask = _raised_cosine_mask(
            rows, cols, cutoff_fraction, rolloff_fraction)
    else:
        raise ValueError(f"Unknown filter_type: {filter_type}")

    F_filtered = F * mask
    img_back = np.fft.ifft2(np.fft.ifftshift(F_filtered))
    img_back = np.real(img_back)

    if should_rescale:
        img_back = np.clip(img_back, 0.0, 1.0)
        return (img_back * 255).astype(np.uint8)
    else:
        # Return in original arbitrary range
        return img_back
