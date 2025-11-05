import os
import json
import time
from datetime import datetime
import imageio
import numpy as np
from scripts.utils import compute_metrics


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_report(frames, report_name=None, out_root="results/reports"):
    """Save captured frames to a named report folder.

    frames: list of dicts with keys: denoised (ndarray), metrics (dict), params (dict), alg, name
    report_name: custom name for the report folder (if None, uses timestamp)
    Returns the output directory path.
    """
    if report_name is None:
        t = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"report_{t}"
    else:
        # Sanitize the report name
        folder_name = "".join(c if c.isalnum() or c in (
            ' ', '_', '-') else '_' for c in report_name)
        folder_name = folder_name.strip()
        if not folder_name:
            folder_name = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    out_dir = os.path.join(out_root, folder_name)
    _ensure_dir(out_dir)
    rows = []

    # Get clean and noisy images from first frame (they're the same for all frames)
    if frames:
        # Assuming frames store references to the same dataset item
        # We need to pass clean/noisy separately or retrieve from dataset
        pass

    for i, f in enumerate(frames, start=1):
        denoised_img = f.get("denoised")
        clean_img = f.get("clean")
        noisy_img = f.get("noisy")

        # Save individual denoised image
        img_name = f"frame_{i}_{f.get('name', 'img')}_{f.get('alg', 'alg')}.png"
        img_path = os.path.join(out_dir, img_name)

        # Ensure image is uint8 [0,255]
        def to_uint8(img):
            if img is None:
                return None
            if img.dtype == np.uint8:
                return img
            if img.max() <= 1.0:
                return (img * 255).astype('uint8')
            return np.clip(img, 0, 255).astype('uint8')

        denoised_uint8 = to_uint8(denoised_img)
        imageio.imwrite(img_path, denoised_uint8)

        # Generate comparison image (Clean | Noisy | Denoised) side by side
        if clean_img is not None and noisy_img is not None:
            clean_uint8 = to_uint8(clean_img)
            noisy_uint8 = to_uint8(noisy_img)

            # Create side-by-side comparison
            comparison = np.hstack([clean_uint8, noisy_uint8, denoised_uint8])

            # Add labels to comparison (optional, using text overlay)
            from PIL import Image, ImageDraw, ImageFont
            comparison_pil = Image.fromarray(comparison)
            draw = ImageDraw.Draw(comparison_pil)

            # Try to use a font, fallback to default
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()

            # Add labels
            h, w = clean_uint8.shape
            draw.text((10, 10), "Clean", fill=255, font=font)
            draw.text((w + 10, 10), "Noisy", fill=255, font=font)
            draw.text((2*w + 10, 10), "Denoised", fill=255, font=font)

            # Save comparison image
            comparison_name = f"comparison_{i}_{f.get('name', 'img')}_{f.get('alg', 'alg')}.png"
            comparison_path = os.path.join(out_dir, comparison_name)
            comparison_pil.save(comparison_path)

        # Calculate noisy image metrics (baseline comparison)
        noisy_metrics = None
        if clean_img is not None and noisy_img is not None:
            noisy_metrics = compute_metrics(clean_img, noisy_img)

        row = {
            "index": i,
            "name": f.get('name'),
            "algorithm": f.get('alg'),
            "image_path": img_path,
            "comparison_path": comparison_path if clean_img is not None else None,
            "metrics": f.get('metrics', {}),
            "noisy_metrics": noisy_metrics,  # Baseline metrics for noisy image
            "params": f.get('params', {})
        }
        rows.append(row)
    # save metadata json
    meta_path = os.path.join(out_dir, "report_meta.json")
    with open(meta_path, 'w', encoding='utf8') as fh:
        json.dump(
            {"frames": rows, "generated": datetime.now().isoformat()}, fh, indent=2)
    return out_dir
