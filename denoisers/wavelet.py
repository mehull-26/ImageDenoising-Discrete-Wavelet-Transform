"""Wavelet denoiser using PyWavelets.

Pure wavelet thresholding for denoising. Best for AWGN and speckle noise
where spatial structure is preserved. Uses adaptive BayesShrink thresholding.

Returns uint8 image in [0,255] range.
"""
import numpy as np
import pywt


def _estimate_noise_sigma(coeffs):
    """Estimate noise sigma from finest scale HH subband using MAD estimator."""
    try:
        # coeffs[-1] contains (cH, cV, cD) - finest detail coefficients
        cH, cV, cD = coeffs[-1]
        # Use diagonal detail (cD/HH) for noise estimation
        sigma = np.median(np.abs(cD)) / 0.6745
        return max(float(sigma), 1e-10)
    except Exception:
        return 1e-10


def denoise(image: np.ndarray, wavelet: str = "sym4", level: int = None, method: str = "bayes") -> np.ndarray:
    """Apply wavelet denoising using soft thresholding.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (uint8 [0,255], float [0,1], or arbitrary range for transforms)
    wavelet : str
        Wavelet basis: 'sym4', 'db4', 'db8', 'coif4' (sym4 recommended for images)
    level : int or None
        Decomposition level (None = auto, typically 3-4 for images)
    method : str
        'bayes' (BayesShrink, adaptive) or 'visu' (VisuShrink, global)

    Returns
    -------
    np.ndarray
        Denoised image in uint8 [0,255]
    """
    # Store if input is uint8 for output
    was_uint8 = image.dtype == np.uint8

    # Normalize to [0,1] float ONLY if uint8 or >1
    if image.dtype == np.uint8:
        img = image.astype(np.float64) / 255.0
        should_rescale_output = True
    elif image.max() > 1.0 or image.min() < 0.0:
        # Arbitrary range (e.g., log domain) - work directly without normalization
        img = image.astype(np.float64)
        should_rescale_output = False
    else:
        img = image.astype(np.float64)
        should_rescale_output = True

    # Only clip if we're in [0,1] range
    if should_rescale_output:
        img = np.clip(img, 0.0, 1.0)

    # Wavelet decomposition
    coeffs = pywt.wavedec2(img, wavelet=wavelet, level=level, mode='symmetric')

    # Estimate noise level from finest scale
    sigma_n = _estimate_noise_sigma(coeffs)

    if method == "bayes":
        # BayesShrink: adaptive threshold per subband
        new_coeffs = [coeffs[0]]  # Keep approximation coefficients

        for detail_level in coeffs[1:]:
            cH, cV, cD = detail_level
            processed = []

            for sub in (cH, cV, cD):
                # Estimate signal variance
                sigma_y = np.std(sub)
                sigma_y_sq = sigma_y ** 2
                sigma_n_sq = sigma_n ** 2

                # BayesShrink threshold formula
                sigma_x_sq = max(sigma_y_sq - sigma_n_sq, 0.0)

                if sigma_x_sq > 0:
                    # Threshold = sigma_n^2 / sigma_x
                    thresh = sigma_n_sq / np.sqrt(sigma_x_sq)
                else:
                    # No signal, aggressive thresholding
                    thresh = sigma_y if sigma_y > 0 else sigma_n

                # Apply soft thresholding
                sub_thresh = pywt.threshold(sub, thresh, mode='soft')
                processed.append(sub_thresh)

            new_coeffs.append(tuple(processed))

        rec = pywt.waverec2(new_coeffs, wavelet=wavelet, mode='symmetric')

    elif method == "visu":
        # VisuShrink: universal threshold (more aggressive)
        # Threshold = sigma * sqrt(2*log(N))
        n = img.size
        thresh = sigma_n * np.sqrt(2.0 * np.log(n))

        # Apply threshold to all detail coefficients
        new_coeffs = [coeffs[0]]
        for detail_level in coeffs[1:]:
            cH, cV, cD = detail_level
            new_coeffs.append((
                pywt.threshold(cH, thresh, mode='soft'),
                pywt.threshold(cV, thresh, mode='soft'),
                pywt.threshold(cD, thresh, mode='soft')
            ))

        rec = pywt.waverec2(new_coeffs, wavelet=wavelet, mode='symmetric')

    else:
        raise ValueError(f"Unknown method: {method}. Use 'bayes' or 'visu'")

    # Match original size (wavelet transform may change size slightly)
    if rec.shape != img.shape:
        rec = rec[:img.shape[0], :img.shape[1]]

    # Convert to uint8 [0,255] only if input was in [0,1] range
    if should_rescale_output:
        rec = np.clip(rec, 0.0, 1.0)
        return (rec * 255).astype(np.uint8)
    else:
        # Return in original arbitrary range (e.g., log domain)
        return rec
