"""Backend wrapper for the interactive UI.

Provides functions to list dataset items, run denoisers, and save results.
All denoisers now return uint8 [0,255] images for consistency.
Applies noise-specific preprocessing (log transform for speckle, Anscombe for Poisson).
"""
from pathlib import Path
import time
import pandas as pd
import numpy as np

from scripts.data_gen import generate_demo_dataset
from scripts.utils import save_image, compute_metrics
from denoisers.gaussian import denoise as gaussian_denoise
from denoisers.fourier import denoise as fourier_denoise
from denoisers.wavelet import denoise as wavelet_denoise
from denoisers.median import denoise as median_denoise
from denoisers.noise_transforms import preprocess_for_noise_type, postprocess_from_noise_type
import itertools
import math


def get_demo_dataset():
    """Return the demo dataset (list of dicts)."""
    return generate_demo_dataset()


def run_denoiser(item: dict, algorithm: str, params: dict):
    """Run the specified denoiser on the provided dataset item.

    Applies noise-specific preprocessing based on item['noise_type']:
    - speckle: log transform (converts multiplicative to additive)
    - poisson: Anscombe transform (stabilizes variance)
    - awgn/sp: no preprocessing

    Parameters
    ----------
    item: dict
        Dataset item with keys: name, clean, noisy, noise_type
    algorithm: str
        'median'|'gaussian'|'fourier'|'wavelet'
    params: dict
        Algorithm-specific parameters

    Returns
    -------
    denoised: np.ndarray (uint8 [0,255])
    metrics: dict (mse, psnr, ssim)
    elapsed: float (seconds)
    """
    alg = algorithm.lower()
    noisy = item.get("noisy")
    clean = item.get("clean")
    noise_type = item.get("noise_type", "awgn")

    # Check if user wants to force log transform (for wavelet/fourier on speckle)
    use_log_transform = params.get("use_log_transform", False)

    # Override noise type preprocessing if log transform is explicitly requested
    if use_log_transform and noise_type == 'speckle':
        # Force log transform for speckle
        from denoisers.noise_transforms import log_transform_simple
        # Ensure noisy is in [0,1] float format
        if isinstance(noisy, np.ndarray):
            if noisy.dtype == np.uint8:
                noisy_float = noisy.astype(np.float64) / 255.0
            else:
                noisy_float = noisy.astype(np.float64)
                if noisy_float.max() > 1.0:
                    noisy_float = noisy_float / 255.0
        else:
            noisy_float = np.array(noisy, dtype=np.float64)

        noisy_float = np.clip(noisy_float, 0.0, 1.0)
        preprocessed, log_min, log_max = log_transform_simple(noisy_float)
        transform_type = (
            'log_simple', {'log_min': log_min, 'log_max': log_max})
    else:
        # Standard preprocessing
        preprocessed, transform_type = preprocess_for_noise_type(
            noisy, noise_type)

    t0 = time.perf_counter()
    if alg == "median":
        size = int(params.get("size", 3))
        denoised_transformed = median_denoise(preprocessed, size=size)
    elif alg == "gaussian":
        sigma = float(params.get("sigma", 1.0))
        denoised_transformed = gaussian_denoise(preprocessed, sigma=sigma)
    elif alg == "fourier":
        cutoff = float(params.get("cutoff_fraction", 0.1))
        filter_type = params.get("filter_type", "gaussian")
        sigma_fraction = params.get("sigma_fraction", None)
        rolloff_fraction = float(params.get("rolloff_fraction", 0.05))
        denoised_transformed = fourier_denoise(preprocessed, cutoff_fraction=cutoff, filter_type=filter_type,
                                               sigma_fraction=sigma_fraction, rolloff_fraction=rolloff_fraction)
    elif alg == "wavelet":
        wavelet = params.get("wavelet", "sym4")
        level = params.get("level", None)
        method = params.get("method", "bayes")
        denoised_transformed = wavelet_denoise(
            preprocessed, wavelet=wavelet, level=level, method=method)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Convert denoiser output to float for inverse transform
    if denoised_transformed.dtype == np.uint8:
        denoised_float = denoised_transformed.astype(np.float64) / 255.0
    else:
        # Already float - use as-is (may be in arbitrary range like log domain)
        denoised_float = denoised_transformed.astype(np.float64)

    # Apply inverse transform
    denoised_original_domain = postprocess_from_noise_type(
        denoised_float, transform_type)

    # Convert back to uint8 for output
    denoised = (np.clip(denoised_original_domain,
                0.0, 1.0) * 255).astype(np.uint8)

    t1 = time.perf_counter()
    elapsed = t1 - t0

    metrics = compute_metrics(clean, denoised) if clean is not None else {}
    return denoised, metrics, elapsed


def save_run(item: dict, denoised, algorithm: str, params: dict, output_root: str = "results"):
    """Save denoised image and append metadata to per-noise CSV.

    Returns path to saved image and CSV path.
    """
    out_root = Path(output_root)
    noise_type = item.get("noise_type", "unknown")
    out_dir = out_root / noise_type
    out_dir.mkdir(parents=True, exist_ok=True)

    name = item.get("name")
    # save images
    save_image(denoised, out_dir / f"{name}_{algorithm}.png")
    save_image(item.get("noisy"), out_dir / f"{name}_noisy.png")
    if item.get("clean") is not None:
        save_image(item.get("clean"), out_dir / f"{name}_clean.png")

    # append to CSV
    csv_path = out_dir / "results_summary.csv"
    metrics = compute_metrics(item.get("clean"), denoised) if item.get(
        "clean") is not None else {}
    record = {**metrics, "image": name,
              "algorithm": algorithm, "noise_type": noise_type}
    for k, v in params.items():
        record[f"param_{k}"] = v

    df = pd.DataFrame([record])
    if csv_path.exists():
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)

    return out_dir / f"{name}_{algorithm}.png", csv_path


def generate_comparison(item: dict, algorithms: list, params_map: dict = None):
    """Run multiple denoisers on `item` and return a DataFrame with metrics and timings.

    Parameters
    ----------
    item: dict
        dataset item
    algorithms: list
        list of algorithm names to run
    params_map: dict
        mapping algorithm -> params dict

    Returns
    -------
    df: pandas.DataFrame with columns [algorithm, mse, psnr, ssim, time_s]
    denoised_images: dict algorithm->image
    """
    records = []
    denoised_images = {}
    params_map = params_map or {}

    for alg in algorithms:
        params = params_map.get(alg, {})
        denoised, metrics, elapsed = run_denoiser(item, alg, params)
        rec = {"algorithm": alg, "mse": metrics.get("mse"), "psnr": metrics.get(
            "psnr"), "ssim": metrics.get("ssim"), "time_s": elapsed}
        # include params flattened
        for k, v in params.items():
            rec[f"param_{k}"] = v
        records.append(rec)
        denoised_images[alg] = denoised

    df = pd.DataFrame.from_records(records)
    return df, denoised_images


def parameter_search(item: dict, algorithm: str, grid: dict, balance_weight=0.6):
    """Brute-force parameter search over `grid` for `algorithm` on `item`.

    Selection criterion: balanced score = balance_weight*normalized_SSIM + (1-balance_weight)*normalized_PSNR.
    SSIM is already in [0,1]; PSNR is normalized as psnr_norm = min(psnr/50, 1) so 50dB+ maps to 1.

    Parameters
    ----------
    item: dict
    algorithm: str
    grid: dict mapping param name to iterable of values
    balance_weight: float (default 0.6)
        Weight for SSIM; PSNR weight = 1 - balance_weight.

    Returns
    -------
    best_params: dict
    best_image: ndarray
    results_df: pandas.DataFrame (one row per tried parameter combination, includes 'score' column)
    """
    # create product of grid
    keys = list(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in keys)))

    records = []
    best_score = -math.inf
    best_image = None
    best_params = None

    for combo in combos:
        params = {k: v for k, v in zip(keys, combo)}
        denoised, metrics, elapsed = run_denoiser(item, algorithm, params)
        rec = {**metrics, "time_s": elapsed}
        rec.update({f"param_{k}": v for k, v in params.items()})

        # compute balanced score
        ssim = rec.get("ssim", 0.0)
        psnr_v = rec.get("psnr", 0.0)
        # normalize psnr: assume 50dB is excellent, cap at 1
        psnr_norm = min(psnr_v / 50.0, 1.0) if psnr_v > 0 else 0.0
        score = balance_weight * ssim + (1.0 - balance_weight) * psnr_norm
        rec["score"] = score
        records.append(rec)

        if score > best_score:
            best_score = score
            best_image = denoised
            best_params = params.copy()

    df = pd.DataFrame.from_records(records)
    return best_params, best_image, df
