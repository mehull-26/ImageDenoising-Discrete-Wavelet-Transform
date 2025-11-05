"""Utility functions: metrics and IO helpers."""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.io import imsave


def _normalize_to_float(img):
    """Normalize image to [0,1] float for metric computation."""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    img = img.astype(np.float32)
    if img.max() > 1.0:
        return img / 255.0
    return img


def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = _normalize_to_float(a)
    b = _normalize_to_float(b)
    return float(np.mean((a - b) ** 2))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = _normalize_to_float(a)
    b = _normalize_to_float(b)
    return float(peak_signal_noise_ratio(a, b, data_range=1.0))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    a = _normalize_to_float(a)
    b = _normalize_to_float(b)
    return float(structural_similarity(a, b, data_range=1.0))


def compute_metrics(clean: np.ndarray, denoised: np.ndarray) -> dict:
    return {"mse": mse(clean, denoised), "psnr": psnr(clean, denoised), "ssim": ssim(clean, denoised)}


def save_image(img: np.ndarray, path):
    """Save image to file. Handles both uint8 and float inputs."""
    if img.dtype == np.uint8:
        imsave(str(path), img)
    else:
        # Normalize to [0,1] and convert to uint8
        arr = _normalize_to_float(img)
        imsave(str(path), (arr * 255).astype('uint8'))
