Image Denoising with Wavelet Transform — Pipeline

This educational project compares three denoising approaches on grayscale images:

- Gaussian spatial denoising (blurring)
- Frequency-domain low-pass filtering (Fourier)
- Wavelet-transform denoising (DWT + thresholding)

Project layout

- denoisers/
  - gaussian.py — Gaussian blur denoiser (implemented)
  - fourier.py — Fourier low-pass denoiser (stub)
  - wavelet.py — Wavelet denoiser (stub)
- scripts/
  - data_gen.py — generate synthetic clean and noisy images
  - utils.py — metrics and helper IO functions
- data/ — generated datasets (clean and noisy)
- results/ — denoised outputs and CSV summaries
- tests/ — unit and integration tests
- main.py — pipeline entrypoint (runs experiments and saves results)
- requirements.txt — Python dependencies

Quick start (after creating a virtual environment and installing requirements):

1. Install dependencies

   python -m pip install -r requirements.txt

2. Run the pipeline (demo):

   python main.py --output-dir results --demo

This will generate noisy images, run the Gaussian denoiser (and placeholder for others), save outputs into `results/`, and write a CSV with metrics and timings.

Notes

- This project focuses on grayscale-only processing for clarity in comparison and presentation.
- All processing times for each algorithm are recorded and saved in the CSV results for use in slides.
