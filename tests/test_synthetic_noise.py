import numpy as np
from skimage import img_as_float
from scripts.utils import psnr
from denoisers.gaussian import denoise as gaussian_denoise
from scripts.data_gen import add_awgn


def synthetic_gradient(size=128):
    # simple horizontal gradient where denoising effects are visible
    x = np.linspace(0, 1, size)
    img = np.tile(x, (size, 1))
    return img


def test_gaussian_improves_psnr():
    img = synthetic_gradient(128)
    noisy = add_awgn(img, sigma=0.1, seed=42)
    den = gaussian_denoise(noisy, sigma=1.0)
    noisy_psnr = psnr(img, noisy)
    den_psnr = psnr(img, den)
    # Expect Gaussian denoising to improve PSNR for AWGN
    assert den_psnr > noisy_psnr
