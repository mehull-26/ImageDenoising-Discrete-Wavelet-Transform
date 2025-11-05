"""Noise-specific preprocessing transforms.

Different noise types require different preprocessing:
- AWGN/SP: Direct denoising (additive noise)
- Speckle: Log-transform (multiplicative noise)
- Poisson: Anscombe transform (signal-dependent noise)
"""
import numpy as np


def log_transform(image: np.ndarray, offset: float = 0.01) -> np.ndarray:
    """Apply log transform for multiplicative (speckle) noise.

    Converts multiplicative noise to additive in log domain.
    Uses base-10 log for better numerical stability.

    Parameters
    ----------
    image: np.ndarray
        Input image in [0, 1] range
    offset: float
        Offset to avoid log(0), default 0.01

    Returns
    -------
    np.ndarray
        Log-transformed image normalized to [0, 1]
    """
    # Add offset and apply log10 (more stable than ln)
    img = np.maximum(image, offset)
    log_img = np.log10(img + offset)

    # Normalize: log10(offset) to log10(1+offset)
    # For offset=0.01: range is approximately [-1.996, 0.004]
    log_min = np.log10(offset)
    log_max = np.log10(1.0 + offset)

    # Normalize to [0, 1]
    normalized = (log_img - log_min) / (log_max - log_min)

    return np.clip(normalized, 0.0, 1.0)


def inverse_log_transform(log_image: np.ndarray, offset: float = 0.01) -> np.ndarray:
    """Inverse log transform to get back original domain.

    Parameters
    ----------
    log_image: np.ndarray
        Log-transformed normalized image in [0, 1]
    offset: float
        Same offset used in forward transform

    Returns
    -------
    np.ndarray
        Image in [0, 1] range
    """
    # Denormalize from [0, 1] back to log range
    log_min = np.log10(offset)
    log_max = np.log10(1.0 + offset)
    log_denorm = log_image * (log_max - log_min) + log_min

    # Apply 10^x and subtract offset
    result = np.power(10.0, log_denorm) - offset
    return np.clip(result, 0.0, 1.0)


def anscombe_transform(image: np.ndarray) -> np.ndarray:
    """Apply Anscombe transform for Poisson noise.

    Converts signal-dependent Poisson noise to approximately Gaussian
    with constant variance. Normalizes to [0, 1] range for processing.

    Parameters
    ----------
    image: np.ndarray
        Input image in [0, 1] range

    Returns
    -------
    np.ndarray
        Anscombe-transformed and normalized image in [0, 1]
    """
    # Anscombe: 2*sqrt(x + 3/8)
    img = np.maximum(image, 0)
    anscombe = 2.0 * np.sqrt(img + 3.0/8.0)

    # Normalize to [0, 1] for denoiser compatibility
    # Min: 2*sqrt(3/8) ≈ 1.225, Max: 2*sqrt(1 + 3/8) ≈ 2.345
    anscombe_min = 2.0 * np.sqrt(3.0/8.0)
    anscombe_max = 2.0 * np.sqrt(1.0 + 3.0/8.0)
    normalized = (anscombe - anscombe_min) / (anscombe_max - anscombe_min)

    return np.clip(normalized, 0.0, 1.0)


def inverse_anscombe_transform(anscombe_image: np.ndarray) -> np.ndarray:
    """Inverse Anscombe transform.

    Parameters
    ----------
    anscombe_image: np.ndarray
        Anscombe-transformed normalized image in [0, 1]

    Returns
    -------
    np.ndarray
        Image in original [0, 1] domain
    """
    # Denormalize from [0, 1] back to Anscombe range
    anscombe_min = 2.0 * np.sqrt(3.0/8.0)
    anscombe_max = 2.0 * np.sqrt(1.0 + 3.0/8.0)
    anscombe_denorm = anscombe_image * \
        (anscombe_max - anscombe_min) + anscombe_min

    # Inverse: (x/2)^2 - 3/8
    result = (anscombe_denorm / 2.0) ** 2 - 3.0/8.0
    return np.clip(result, 0.0, 1.0)


def preprocess_for_noise_type(image: np.ndarray, noise_type: str) -> tuple:
    """Preprocess image based on noise type.

    Parameters
    ----------
    image: np.ndarray
        Input noisy image (uint8 [0,255] or float [0,1])
    noise_type: str
        'awgn', 'sp', 'speckle', 'poisson'

    Returns
    -------
    tuple: (preprocessed_image, transform_metadata)
        preprocessed_image: Image ready for denoising [0,1]
        transform_metadata: Either 'none' (str) or (transform_type, params_dict) (tuple)
    """
    # Normalize to [0,1]
    if image.dtype == np.uint8:
        img = image.astype(np.float64) / 255.0
    else:
        img = image.astype(np.float64)
        if img.max() > 1.0:
            img = img / 255.0

    img = np.clip(img, 0.0, 1.0)

    noise_type = noise_type.lower()

    # Apply appropriate transform based on noise model
    if noise_type == 'speckle':
        # Speckle: DO NOT use transforms - denoise directly
        # Multiplicative model requires specialized filters (Lee, Frost)
        # Standard denoisers work better on raw speckle than on log-transformed
        return img, 'none'
    elif noise_type == 'poisson':
        # Poisson is signal-dependent - use Anscombe
        return anscombe_transform(img), 'anscombe'
    else:
        # AWGN, SP: additive noise, no preprocessing
        return img, 'none'


def log_transform_simple(image: np.ndarray) -> tuple:
    """Log transform for speckle (multiplicative) noise using robust normalization.

    Uses percentile-based normalization with padding to avoid extreme compression.
    This preserves noise structure while making it suitable for denoisers.
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    img = np.maximum(image, eps)

    # Apply natural log
    log_img = np.log(img + eps)

    # Use robust percentile-based normalization with padding
    # This prevents outliers from compressing the main data range
    p1 = np.percentile(log_img, 0.5)   # 0.5th percentile
    p99 = np.percentile(log_img, 99.5)  # 99.5th percentile

    # Add 10% padding to avoid edge artifacts
    log_range = p99 - p1
    log_min = p1 - 0.1 * log_range
    log_max = p99 + 0.1 * log_range

    # Normalize to [0, 1]
    if log_max - log_min > eps:
        log_normalized = (log_img - log_min) / (log_max - log_min)
        log_normalized = np.clip(log_normalized, 0.0, 1.0)
    else:
        log_normalized = np.zeros_like(log_img)

    return log_normalized, log_min, log_max


def inverse_log_transform_simple(log_normalized: np.ndarray, log_min: float, log_max: float) -> np.ndarray:
    """Inverse of log transform.

    Parameters
    ----------
    log_normalized : Denoised image in normalized log domain [0,1]
    log_min : Minimum log value used for normalization
    log_max : Maximum log value used for normalization
    """
    # Denormalize from [0,1] back to log range
    log_img = log_normalized * (log_max - log_min) + log_min

    # Apply exp
    result = np.exp(log_img)
    return np.clip(result, 0.0, 1.0)


def postprocess_from_noise_type(image: np.ndarray, transform_metadata) -> np.ndarray:
    """Apply inverse transform based on preprocessing type.

    Parameters
    ----------
    image: np.ndarray
        Denoised image in transformed domain
    transform_metadata: str or tuple
        If str: 'none', 'log', 'anscombe'
        If tuple: (transform_type, params_dict) e.g. ('log_simple', {'log_min': ..., 'log_max': ...})

    Returns
    -------
    np.ndarray
        Image in [0, 1] range
    """
    # Handle string format (backward compatibility)
    if isinstance(transform_metadata, str):
        if transform_metadata == 'log':
            return inverse_log_transform(image, offset=0.01)
        elif transform_metadata == 'anscombe':
            return inverse_anscombe_transform(image)
        else:
            return np.clip(image, 0.0, 1.0)

    # Handle tuple format: (transform_type, params_dict)
    transform_type, params = transform_metadata

    if transform_type == 'log_simple':
        log_min = params['log_min']
        log_max = params['log_max']
        return inverse_log_transform_simple(image, log_min, log_max)
    else:
        return np.clip(image, 0.0, 1.0)
