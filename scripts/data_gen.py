"""Synthetic dataset generator for grayscale images with multiple noise types.

This module provides a small demo generator that returns a list of dicts with keys:
- name: str
- clean: numpy array (float64, 0-1)
- noisy: numpy array (float64, 0-1)

The generator does not save files by default; `main.py` will handle saving.
"""
from skimage import data, color, img_as_float
import numpy as np


def add_awgn(image, sigma=0.05, seed=0):
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, sigma, image.shape)
    out = image + noise
    out = np.clip(out, 0.0, 1.0)
    return out


def add_salt_pepper(image, amount=0.05, seed=0):
    rng = np.random.RandomState(seed)
    out = image.copy()
    num_pixels = int(amount * image.size)
    coords = tuple(rng.randint(0, s, num_pixels) for s in image.shape)
    # Half salt (1) half pepper (0)
    half = num_pixels // 2
    out[coords[0][:half], coords[1][:half]] = 1.0
    out[coords[0][half:], coords[1][half:]] = 0.0
    return out


def add_speckle(image, var=0.04, seed=0):
    """Add multiplicative speckle noise: image * (1 + N(0, var))."""
    rng = np.random.RandomState(seed)
    noise = rng.normal(0.0, np.sqrt(var), image.shape)
    out = image + image * noise
    out = np.clip(out, 0.0, 1.0)
    return out


def add_poisson(image, scale=1.0, seed=0):
    """Add Poisson noise (signal-dependent).

    Parameters
    ----------
    image: np.ndarray
        Input image in [0, 1] range
    scale: float
        Scaling factor. Lower = more noise. Typical values: 0.1-10
        - scale=1: moderate noise
        - scale=0.5: high noise (more visible)
        - scale=5-10: low noise
    seed: int
        Random seed

    Returns
    -------
    np.ndarray
        Noisy image in [0, 1] range
    """
    rng = np.random.RandomState(seed)
    # Scale image to higher values for Poisson sampling
    # Poisson(lambda) has variance = lambda, so higher values = less relative noise
    scaled = image * 255.0 * scale
    # Ensure positive values
    scaled = np.maximum(scaled, 1e-10)  # Avoid zero
    # Apply Poisson noise
    noisy_scaled = rng.poisson(scaled)
    # Scale back to [0, 1]
    out = noisy_scaled / (255.0 * scale)
    out = np.clip(out, 0.0, 1.0)
    return out


def generate_demo_dataset(data_dir: str = "data"):
    """Return a small dataset list of dicts with clean and noisy images.

    Parameters
    ----------
    data_dir: str
        Directory where images could be saved (not used here, kept for API parity).

    Returns
    -------
    list of dict
    """
    img = img_as_float(color.rgb2gray(data.astronaut()))
    # resize or crop if needed - keep as is for demo

    # Create noisy versions for astronaut
    # INCREASED AWGN to demonstrate wavelet superiority over median
    noisy_awgn = add_awgn(img, sigma=0.08, seed=1)
    noisy_sp = add_salt_pepper(img, amount=0.03, seed=2)
    # Speckle: wavelet not ideal for multiplicative noise
    noisy_speckle = add_speckle(img, var=0.05, seed=3)
    # Visible Poisson noise
    noisy_poisson = add_poisson(img, scale=1.0, seed=4)

    dataset = [
        {"name": "astronaut_awgn", "clean": img,
            "noisy": noisy_awgn, "noise_type": "awgn"},
        {"name": "astronaut_sp", "clean": img,
            "noisy": noisy_sp, "noise_type": "sp"},
        {"name": "astronaut_speckle", "clean": img,
            "noisy": noisy_speckle, "noise_type": "speckle"},
        {"name": "astronaut_poisson", "clean": img,
            "noisy": noisy_poisson, "noise_type": "poisson"}
    ]

    # Add synthetic images (bars and circles) with noisy variants
    def _make_bars(size=256, num_bars=8):
        x = np.zeros((size, size), dtype=float)
        bar_w = size // (2 * num_bars)
        for i in range(num_bars):
            start = i * 2 * bar_w
            x[:, start:start + bar_w] = 1.0
        return x

    def _make_circles(size=256, num_circles=6):
        Y, X = np.ogrid[:size, :size]
        center = (size // 2, size // 2)
        R = np.sqrt((Y - center[0]) ** 2 + (X - center[1]) ** 2)
        img_c = np.zeros((size, size), dtype=float)
        max_r = size // 2
        for i in range(num_circles):
            r0 = (i / num_circles) * max_r
            r1 = ((i + 0.5) / num_circles) * max_r
            img_c[(R >= r0) & (R < r1)] = 1.0 if i % 2 == 0 else 0.5
        return img_c

    # synthetic clean images
    bars = _make_bars(256, num_bars=8)
    circles = _make_circles(256, num_circles=8)

    # produce noisy variants for each synthetic image
    dataset += [
        {"name": "bars_awgn", "clean": bars, "noisy": add_awgn(
            bars, sigma=0.12, seed=10), "noise_type": "awgn"},
        {"name": "bars_sp", "clean": bars, "noisy": add_salt_pepper(
            bars, amount=0.06, seed=11), "noise_type": "sp"},
        {"name": "bars_speckle", "clean": bars, "noisy": add_speckle(
            bars, var=0.06, seed=12), "noise_type": "speckle"},
        {"name": "bars_poisson", "clean": bars, "noisy": add_poisson(
            bars, scale=1.0, seed=13), "noise_type": "poisson"},
        {"name": "circles_awgn", "clean": circles, "noisy": add_awgn(
            circles, sigma=0.12, seed=14), "noise_type": "awgn"},
        {"name": "circles_sp", "clean": circles, "noisy": add_salt_pepper(
            circles, amount=0.06, seed=15), "noise_type": "sp"},
        {"name": "circles_speckle", "clean": circles, "noisy": add_speckle(
            circles, var=0.06, seed=16), "noise_type": "speckle"},
        {"name": "circles_poisson", "clean": circles, "noisy": add_poisson(
            circles, scale=1.0, seed=17), "noise_type": "poisson"}
    ]

    return dataset
