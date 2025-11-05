"""Main pipeline for running image denoising experiments (grayscale only).

Usage:
    python main.py --output-dir results --demo

The demo mode generates a small dataset and runs configured denoisers to produce outputs, CSV metrics and timing.
"""
import argparse
import os
from pathlib import Path
import time
import pandas as pd
import yaml

from scripts.data_gen import generate_demo_dataset
from scripts.utils import save_image, compute_metrics

# Import denoisers
from denoisers.gaussian import denoise as gaussian_denoise
from denoisers.fourier import denoise as fourier_denoise
from denoisers.wavelet import denoise as wavelet_denoise
from denoisers.median import denoise as median_denoise


def load_config(path: str = "config.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pipeline(output_dir: str, demo: bool = False, config_path: str = "config.yaml"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(config_path)
    denoisers_cfg = cfg.get("denoisers", ["gaussian"]) if cfg else ["gaussian"]

    # read denoiser params with sensible defaults
    median_cfg = cfg.get("median", {}) if cfg else {}
    gaussian_cfg = cfg.get("gaussian", {}) if cfg else {}
    fourier_cfg = cfg.get("fourier", {}) if cfg else {}
    wavelet_cfg = cfg.get("wavelet", {}) if cfg else {}

    records = []

    if demo:
        print("Generating demo dataset...")
        dataset = generate_demo_dataset("data")
    else:
        dataset = []
        for p in Path("data").glob("*.png"):
            dataset.append({"name": p.stem, "clean_path": str(p)})

    for item in dataset:
        name = item["name"]
        clean = item.get("clean")    # numpy array
        noisy = item.get("noisy")

        # save noisy and clean into per-noise subfolder
        noise_type = item.get("noise_type", "unknown")
        noise_dir = output_dir / noise_type
        noise_dir.mkdir(parents=True, exist_ok=True)
        save_image(noisy, noise_dir / f"{name}_noisy.png")
        if clean is not None:
            save_image(clean, noise_dir / f"{name}_clean.png")

        for alg in denoisers_cfg:
            alg = alg.lower()
            print(f"Processing {name} with {alg} denoiser...")
            t0 = time.perf_counter()
            if alg == "gaussian":
                sigma = float(gaussian_cfg.get("sigma", 1.0))
                denoised = gaussian_denoise(noisy, sigma=sigma)
                params = {"sigma": sigma}

            elif alg == "median":
                size = int(median_cfg.get("size", 3))
                denoised = median_denoise(noisy, size=size)
                params = {"size": size}

            elif alg == "fourier":
                cutoff = float(fourier_cfg.get("cutoff_fraction", 0.1))
                ftype = str(fourier_cfg.get("type", "ideal"))
                sigma_frac = fourier_cfg.get("sigma_fraction", None)
                rolloff = float(fourier_cfg.get("rolloff_fraction", 0.05))
                denoised = fourier_denoise(
                    noisy, cutoff_fraction=cutoff, filter_type=ftype, sigma_fraction=sigma_frac, rolloff_fraction=rolloff)
                params = {"cutoff_fraction": cutoff, "filter_type": ftype,
                          "sigma_fraction": sigma_frac, "rolloff_fraction": rolloff}

            elif alg == "wavelet":
                wavelet_name = wavelet_cfg.get("wavelet", "db1")
                level = wavelet_cfg.get("level", None)
                method = wavelet_cfg.get("method", "visu")
                denoised = wavelet_denoise(
                    noisy, wavelet=wavelet_name, level=level, method=method)
                params = {"wavelet": wavelet_name,
                          "level": level, "method": method}

            else:
                print(f"Unknown denoiser '{alg}', skipping")
                continue

            t1 = time.perf_counter()
            elapsed = t1 - t0

            # Save denoised image into per-noise folder
            save_image(denoised, noise_dir / f"{name}_{alg}.png")

            # Compute metrics (if clean available)
            metrics = compute_metrics(
                clean, denoised) if clean is not None else {}
            record = {**metrics, "image": name,
                      "algorithm": alg, "time_s": elapsed}
            # include noise_type for grouping
            record["noise_type"] = noise_type
            # attach simple params as strings
            for k, v in params.items():
                record[f"param_{k}"] = v
            records.append(record)

    # Save CSV
    df = pd.DataFrame.from_records(records)
    # Save CSV per-noise type: group records by noise_type
    if records:
        # ensure we include noise_type in records (may not exist for non-demo)
        for r in records:
            if "noise_type" not in r:
                r["noise_type"] = item.get("noise_type", "unknown")
        # group and write
        grouped = {}
        for r in records:
            nt = r.get("noise_type", "unknown")
            grouped.setdefault(nt, []).append(r)

        for nt, recs in grouped.items():
            df_nt = pd.DataFrame.from_records(recs)
            csv_path = output_dir / nt / "results_summary.csv"
            df_nt.to_csv(csv_path, index=False)
            print(f"Saved results to {csv_path}")
    else:
        print("No records to save.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results",
                        help="Directory to save outputs")
    parser.add_argument("--demo", action="store_true",
                        help="Run demo dataset generation")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config YAML file")
    args = parser.parse_args()

    run_pipeline(args.output_dir, demo=args.demo, config_path=args.config)
