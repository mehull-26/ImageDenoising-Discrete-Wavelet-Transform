"""Tkinter desktop GUI for Image Denoising.

Windows-native UI with sticky controls, no scrolling needed.
Features colored status bar instead of popup messages.
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import threading
import queue
from PIL import Image, ImageTk
import numpy as np
import ui.backend as backend
import ui.reporting as reporting
import sys
import os
import io
import tempfile
from pathlib import Path

# Add project root to path FIRST
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

os.chdir(str(project_root))

# Standard library imports

# Third-party imports

# Project imports (AFTER sys.path modification)


class DenoisingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Denoising — Interactive Desktop UI")
        self.root.geometry("1400x900")
        self.root.resizable(True, True)

        self.dataset = backend.get_demo_dataset()
        self.current_item = self.dataset[0] if self.dataset else None
        self.last_run = None
        self.captured_frames = []

        # Cancel flag for long operations
        self.cancel_flag = threading.Event()
        self.running_thread = None

        # Queue for thread-safe UI updates
        self.queue = queue.Queue()

        self._build_ui()
        self._populate_demo_list()
        self._update_params_for_algorithm()
        self.root.after(100, self._process_queue)

    def _build_ui(self):
        # Top menu bar: Demo select + Help
        top_frame = tk.Frame(self.root, bg="#2e2e2e", height=50)
        top_frame.pack(side=tk.TOP, fill=tk.X)
        top_frame.pack_propagate(False)

        tk.Label(top_frame, text="Demo Image:", bg="#2e2e2e", fg="white", font=("Arial", 10)).pack(
            side=tk.LEFT, padx=10)
        self.demo_combo = ttk.Combobox(top_frame, state="readonly", width=30)
        self.demo_combo.pack(side=tk.LEFT, padx=5)
        self.demo_combo.bind("<<ComboboxSelected>>", self._on_demo_select)

        tk.Button(top_frame, text="Help", command=self._show_help, bg="#555", fg="white").pack(
            side=tk.RIGHT, padx=10)

        # Main area: Left params panel + center images
        main_frame = tk.Frame(self.root)
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left params panel (fixed width, scrollable)
        left_canvas = tk.Canvas(main_frame, width=280, bg="#1e1e1e")
        left_canvas.pack(side=tk.LEFT, fill=tk.Y)
        left_scrollbar = tk.Scrollbar(
            main_frame, orient=tk.VERTICAL, command=left_canvas.yview)
        left_scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        left_canvas.configure(yscrollcommand=left_scrollbar.set)

        self.left_frame = tk.Frame(left_canvas, bg="#1e1e1e")
        left_canvas.create_window((0, 0), window=self.left_frame, anchor='nw')
        self.left_frame.bind("<Configure>", lambda e: left_canvas.configure(
            scrollregion=left_canvas.bbox("all")))

        tk.Label(self.left_frame, text="Parameters", bg="#1e1e1e", fg="white", font=(
            "Arial", 12, "bold")).pack(pady=10)

        # Algorithm selector
        tk.Label(self.left_frame, text="Algorithm:", bg="#1e1e1e", fg="white").pack(
            anchor='w', padx=10, pady=5)
        self.alg_var = tk.StringVar(value="wavelet")
        alg_combo = ttk.Combobox(self.left_frame, textvariable=self.alg_var, state="readonly", values=[
                                 "median", "gaussian", "fourier", "wavelet"], width=25)
        alg_combo.pack(padx=10, pady=5)
        alg_combo.bind("<<ComboboxSelected>>",
                       lambda e: self._update_params_for_algorithm())

        # Params container (will be rebuilt per algorithm)
        self.params_container = tk.Frame(self.left_frame, bg="#1e1e1e")
        self.params_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Center: images (Clean, Noisy, Denoised) side by side
        center_frame = tk.Frame(main_frame, bg="#2e2e2e")
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        img_row = tk.Frame(center_frame, bg="#2e2e2e")
        img_row.pack(fill=tk.BOTH, expand=True)

        self.img_labels = {}
        for idx, title in enumerate(["Clean", "Noisy", "Denoised"]):
            col = tk.Frame(img_row, bg="#2e2e2e")
            col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            tk.Label(col, text=title, bg="#2e2e2e", fg="white", font=(
                "Arial", 12, "bold")).pack(pady=5)
            lbl = tk.Label(col, bg="#1e1e1e")
            lbl.pack(fill=tk.BOTH, expand=True)
            self.img_labels[title] = lbl

            # Add copy button below each image
            tk.Button(col, text=f"Copy {title}", command=lambda t=title: self._copy_image(t),
                      bg="#555555", fg="white", font=("Arial", 9), width=12).pack(pady=5)

        # Metrics display (below Denoised)
        self.metrics_label = tk.Label(center_frame, text="Metrics: N/A", bg="#2e2e2e", fg="white", font=(
            "Arial", 10), justify=tk.LEFT, anchor='w')
        self.metrics_label.pack(fill=tk.X, padx=10, pady=5)

        # Status bar (colored progress indicator)
        self.status_frame = tk.Frame(self.root, bg="#2e2e2e", height=40)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_frame.pack_propagate(False)

        self.status_label = tk.Label(self.status_frame, text="● Ready", bg="#2e2e2e",
                                     fg="#00ff00", font=("Arial", 10, "bold"), anchor='w')
        self.status_label.pack(side=tk.LEFT, padx=20, pady=10)

        # Bottom control bar: Run, Auto-tune, Cancel, Capture, Create Report, Save Report
        bottom_frame = tk.Frame(self.root, bg="#2e2e2e", height=60)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)
        bottom_frame.pack_propagate(False)

        tk.Button(bottom_frame, text="Run", command=self._run_denoiser, bg="#007acc", fg="white", font=(
            "Arial", 10, "bold"), width=10).pack(side=tk.LEFT, padx=10, pady=10)
        tk.Button(bottom_frame, text="Auto-tune", command=self._auto_tune, bg="#0078d4", fg="white", font=(
            "Arial", 10, "bold"), width=10).pack(side=tk.LEFT, padx=5, pady=10)
        self.cancel_btn = tk.Button(bottom_frame, text="Cancel", command=self._cancel_operation, bg="#ff4444", fg="white", font=(
            "Arial", 10, "bold"), width=10, state=tk.DISABLED)
        self.cancel_btn.pack(side=tk.LEFT, padx=5, pady=10)
        tk.Button(bottom_frame, text="Capture", command=self._capture_frame, bg="#00a86b", fg="white", font=(
            "Arial", 10, "bold"), width=10).pack(side=tk.LEFT, padx=5, pady=10)
        tk.Button(bottom_frame, text="Clear", command=self._clear_frames, bg="#6c757d", fg="white", font=(
            "Arial", 10, "bold"), width=10).pack(side=tk.LEFT, padx=5, pady=10)
        tk.Button(bottom_frame, text="Report", command=self._create_report, bg="#ff8c00", fg="white", font=(
            "Arial", 10, "bold"), width=10).pack(side=tk.LEFT, padx=5, pady=10)
        tk.Button(bottom_frame, text="Save", command=self._save_report, bg="#d9534f", fg="white", font=(
            "Arial", 10, "bold"), width=10).pack(side=tk.LEFT, padx=5, pady=10)

    def _populate_demo_list(self):
        names = [d["name"] for d in self.dataset]
        self.demo_combo['values'] = names
        if names:
            self.demo_combo.current(0)
            self._on_demo_select(None)

    def _on_demo_select(self, event):
        sel = self.demo_combo.get()
        self.current_item = next(
            (d for d in self.dataset if d["name"] == sel), self.dataset[0])
        self._update_images()

    def _update_images(self):
        if not self.current_item:
            return
        # Display Clean and Noisy
        self._display_image(self.current_item.get("clean"), "Clean")
        self._display_image(self.current_item.get("noisy"), "Noisy")
        # Display last denoised if available
        if self.last_run:
            self._display_image(self.last_run.get("denoised"), "Denoised")
            self._update_metrics_display()
        else:
            self.img_labels["Denoised"].config(image='', text="No result yet")
            self.metrics_label.config(text="Metrics: N/A")

    def _display_image(self, img_array, title):
        if img_array is None:
            self.img_labels[title].config(image='', text="N/A")
            return
        # All denoisers now return uint8 [0,255]
        img = img_array.copy()
        if img.dtype != np.uint8:
            # Convert float [0,1] to uint8 if needed
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)

        pil_img = Image.fromarray(img)
        # Resize to fit label (max 400x400)
        pil_img.thumbnail((400, 400), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil_img)
        self.img_labels[title].config(image=tk_img, text='')
        self.img_labels[title].image = tk_img  # keep reference

    def _update_metrics_display(self):
        if not self.last_run:
            self.metrics_label.config(text="Metrics: N/A")
            return
        metrics = self.last_run.get("metrics", {})
        elapsed = self.last_run.get("elapsed", 0.0)
        psnr_v = metrics.get("psnr")
        ssim_v = metrics.get("ssim")
        psnr_str = f"{psnr_v:.2f} dB" if isinstance(
            psnr_v, (int, float)) else "N/A"
        ssim_str = f"{ssim_v:.3f}" if isinstance(
            ssim_v, (int, float)) else "N/A"
        self.metrics_label.config(
            text=f"PSNR: {psnr_str} | SSIM: {ssim_str} | Time: {elapsed:.3f} s")

    def _set_status(self, message, color="#00ff00"):
        """Update status bar with colored indicator."""
        self.status_label.config(text=f"● {message}", fg=color)
        self.root.update_idletasks()

    def _update_params_for_algorithm(self):
        # Clear old params
        for w in self.params_container.winfo_children():
            w.destroy()

        alg = self.alg_var.get()
        if alg == "median":
            self._build_median_params()
        elif alg == "gaussian":
            self._build_gaussian_params()
        elif alg == "fourier":
            self._build_fourier_params()
        elif alg == "wavelet":
            self._build_wavelet_params()

    def _build_median_params(self):
        tk.Label(self.params_container, text="Median size:", bg="#1e1e1e", fg="white").pack(
            anchor='w', pady=5)
        self.median_size_var = tk.IntVar(value=3)
        tk.Spinbox(self.params_container, from_=1, to=51, increment=1,
                   textvariable=self.median_size_var, width=20).pack(pady=5)

    def _build_gaussian_params(self):
        tk.Label(self.params_container, text="Sigma:", bg="#1e1e1e", fg="white").pack(
            anchor='w', pady=5)
        self.gaussian_sigma_var = tk.DoubleVar(value=1.5)
        tk.Scale(self.params_container, from_=0.1, to=10.0, resolution=0.1,
                 variable=self.gaussian_sigma_var, orient=tk.HORIZONTAL, bg="#1e1e1e", fg="white").pack(fill=tk.X, pady=5)

    def _build_fourier_params(self):
        tk.Label(self.params_container, text="Cutoff fraction:", bg="#1e1e1e", fg="white").pack(
            anchor='w', pady=5)
        self.fourier_cutoff_var = tk.DoubleVar(value=0.1)
        tk.Scale(self.params_container, from_=0.01, to=0.5, resolution=0.01,
                 variable=self.fourier_cutoff_var, orient=tk.HORIZONTAL, bg="#1e1e1e", fg="white").pack(fill=tk.X, pady=5)
        tk.Label(self.params_container, text="Filter type:", bg="#1e1e1e", fg="white").pack(
            anchor='w', pady=5)
        self.fourier_filter_var = tk.StringVar(value="gaussian")
        ttk.Combobox(self.params_container, textvariable=self.fourier_filter_var, state="readonly",
                     values=["ideal", "gaussian", "raised_cosine"], width=20).pack(pady=5)

    def _build_wavelet_params(self):
        tk.Label(self.params_container, text="Wavelet basis:", bg="#1e1e1e", fg="white").pack(
            anchor='w', pady=5)
        self.wavelet_wavelet_var = tk.StringVar(value="sym4")
        ttk.Combobox(self.params_container, textvariable=self.wavelet_wavelet_var, state="readonly",
                     values=["db1", "db4", "db8", "sym4", "sym8", "coif4"], width=20).pack(pady=5)
        tk.Label(self.params_container, text="Decomposition level:", bg="#1e1e1e", fg="white").pack(
            anchor='w', pady=5)
        self.wavelet_level_var = tk.IntVar(value=3)
        tk.Spinbox(self.params_container, from_=1, to=6,
                   textvariable=self.wavelet_level_var, width=20).pack(pady=5)
        tk.Label(self.params_container, text="Thresholding method:", bg="#1e1e1e", fg="white").pack(
            anchor='w', pady=5)
        self.wavelet_method_var = tk.StringVar(value="bayes")
        ttk.Combobox(self.params_container, textvariable=self.wavelet_method_var, state="readonly",
                     values=["bayes", "visu"], width=20).pack(pady=5)

        # Add help text
        help_text = tk.Label(self.params_container,
                             text="BayesShrink: adaptive (best for most noise)\nVisuShrink: aggressive (removes more)\ndb1/Haar: simple, fast\nsym4/sym8: smooth (best for images)",
                             bg="#1e1e1e", fg="#888", font=("Arial", 8), justify=tk.LEFT)
        help_text.pack(anchor='w', pady=5)

    def _get_current_params(self):
        alg = self.alg_var.get()
        if alg == "median":
            return {"size": self.median_size_var.get()}
        elif alg == "gaussian":
            return {"sigma": self.gaussian_sigma_var.get()}
        elif alg == "fourier":
            p = {"cutoff_fraction": self.fourier_cutoff_var.get(
            ), "filter_type": self.fourier_filter_var.get()}
            if p["filter_type"] == "gaussian":
                p["sigma_fraction"] = 0.05
            elif p["filter_type"] == "raised_cosine":
                p["rolloff_fraction"] = 0.05
            return p
        elif alg == "wavelet":
            return {
                "wavelet": self.wavelet_wavelet_var.get(),
                "level": self.wavelet_level_var.get(),
                "method": self.wavelet_method_var.get()
            }
        return {}

    def _cancel_operation(self):
        """Cancel the currently running operation."""
        self.cancel_flag.set()
        self._set_status("Cancelling...", "#ff8800")
        self.cancel_btn.config(state=tk.DISABLED)

    def _run_denoiser(self):
        if not self.current_item:
            self._set_status("Error: No image selected", "#ff0000")
            return
        alg = self.alg_var.get()
        params = self._get_current_params()

        self._set_status("Running denoising...", "#00aaff")
        self.cancel_flag.clear()
        self.cancel_btn.config(state=tk.NORMAL)

        # Run in background thread
        def worker():
            try:
                denoised, metrics, elapsed = backend.run_denoiser(
                    self.current_item, alg, params)
                if not self.cancel_flag.is_set():
                    self.queue.put(
                        ("run_done", (denoised, metrics, elapsed, params, alg)))
                else:
                    self.queue.put(("cancelled", None))
            except Exception as e:
                self.queue.put(("error", str(e)))

        self.running_thread = threading.Thread(target=worker, daemon=True)
        self.running_thread.start()

    def _auto_tune(self):
        if not self.current_item:
            self._set_status("Error: No image selected", "#ff0000")
            return
        alg = self.alg_var.get()
        params = self._get_current_params()

        self._set_status("Auto-tuning (this may take a while)...", "#ff8800")
        self.cancel_flag.clear()
        self.cancel_btn.config(state=tk.NORMAL)

        # Build comprehensive grid for search
        grid = {}
        if alg == "median":
            size = params.get("size", 3)
            # Test range around current size
            grid["size"] = list(range(max(1, size-4), min(21, size+5), 2))
        elif alg == "gaussian":
            sigma = params.get("sigma", 1.5)
            # Test multiple sigma values centered around current
            grid["sigma"] = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
        elif alg == "fourier":
            # Test different cutoff fractions with both filters
            grid = {
                "cutoff_fraction": [0.05, 0.08, 0.1, 0.12, 0.15, 0.2],
                "filter_type": ["gaussian", "raised_cosine"]
            }
        elif alg == "wavelet":
            # Comprehensive wavelet grid - focus on better performing options
            grid = {
                "wavelet": ["db1", "sym4", "sym8", "db4"],
                "level": [3, 4, 5, 6],
                "method": ["bayes"]
            }

        balance = 0.6

        def worker():
            try:
                results = []
                from itertools import product

                # Generate all combinations
                keys = list(grid.keys())
                combos = list(product(*(grid[k] for k in keys)))
                total = len(combos)

                for idx, combo in enumerate(combos):
                    if self.cancel_flag.is_set():
                        self.queue.put(("cancelled", None))
                        return

                    test_params = {k: v for k, v in zip(keys, combo)}
                    try:
                        denoised, metrics, elapsed = backend.run_denoiser(
                            self.current_item, alg, test_params)

                        # Compute balanced score
                        ssim_v = metrics.get("ssim", 0.0)
                        psnr_v = metrics.get("psnr", 0.0)
                        psnr_norm = min(
                            psnr_v / 50.0, 1.0) if psnr_v > 0 else 0.0
                        score = balance * ssim_v + (1.0 - balance) * psnr_norm

                        results.append({
                            "params": test_params.copy(),
                            "image": denoised,
                            "psnr": psnr_v,
                            "ssim": ssim_v,
                            "score": score,
                            "elapsed": elapsed
                        })

                        # Update progress
                        progress = f"Auto-tuning... {idx+1}/{total}"
                        self.queue.put(("progress", progress))
                    except Exception:
                        continue

                if not self.cancel_flag.is_set() and results:
                    # Sort by score descending
                    results.sort(key=lambda x: x["score"], reverse=True)
                    # Take top 10 results
                    top_results = results[:10]
                    self.queue.put(("autotune_done", (alg, top_results)))
                elif not results:
                    self.queue.put(
                        ("error", "Auto-tune found no valid results"))

            except Exception as e:
                self.queue.put(("error", str(e)))

        self.running_thread = threading.Thread(target=worker, daemon=True)
        self.running_thread.start()

    def _copy_image(self, image_type):
        """Copy the specified image to clipboard."""
        try:
            import io

            # Get the appropriate image array
            if image_type == "Clean":
                img_array = self.current_item.get("clean")
            elif image_type == "Noisy":
                img_array = self.current_item.get("noisy")
            elif image_type == "Denoised":
                if not self.last_run:
                    self._set_status("Error: No denoised image yet", "#ff0000")
                    return
                img_array = self.last_run.get("denoised")
            else:
                self._set_status("Error: Invalid image type", "#ff0000")
                return

            if img_array is None:
                self._set_status(
                    f"Error: {image_type} image not available", "#ff0000")
                return

            # Convert to uint8 if needed
            img = img_array.copy()
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = np.clip(img, 0, 255).astype(np.uint8)

            # Convert to PIL Image and copy to clipboard
            pil_img = Image.fromarray(img)

            # Windows clipboard copy
            output = io.BytesIO()
            pil_img.convert('RGB').save(output, 'BMP')
            data = output.getvalue()[14:]  # Remove BMP header for clipboard
            output.close()

            import win32clipboard
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
            win32clipboard.CloseClipboard()

            self._set_status(
                f"{image_type} image copied to clipboard", "#00ff00")
        except ImportError:
            # Fallback: save to temp file and show message
            import tempfile
            import os

            img_array = None
            if image_type == "Clean":
                img_array = self.current_item.get("clean")
            elif image_type == "Noisy":
                img_array = self.current_item.get("noisy")
            elif image_type == "Denoised":
                img_array = self.last_run.get(
                    "denoised") if self.last_run else None

            if img_array is None:
                self._set_status(f"{image_type} not available", "#ff8800")
                return

            # Convert to uint8
            img = img_array.copy()
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = np.clip(img, 0, 255).astype(np.uint8)

            # Save to temp file
            temp_path = os.path.join(
                tempfile.gettempdir(), f"{image_type}_temp.png")
            pil_img = Image.fromarray(img)
            pil_img.save(temp_path)

            self._set_status(f"{image_type} saved to {temp_path}", "#00ff00")
        except Exception as e:
            self._set_status(f"Copy error: {str(e)[:40]}", "#ff0000")

    def _capture_frame(self):
        if not self.last_run:
            self._set_status("Error: Run denoising first", "#ff0000")
            return

        # Include clean and noisy images in the captured frame
        frame_data = self.last_run.copy()
        frame_data['clean'] = self.current_item.get('clean')
        frame_data['noisy'] = self.current_item.get('noisy')

        self.captured_frames.append(frame_data)
        self._set_status(
            f"Frame #{len(self.captured_frames)} captured", "#00ff00")

    def _clear_frames(self):
        """Clear all captured frames."""
        if not self.captured_frames:
            self._set_status("No frames to clear", "#ff8800")
            return

        # Ask for confirmation
        from tkinter import messagebox
        confirm = messagebox.askyesno(
            "Clear Frames",
            f"Are you sure you want to clear all {len(self.captured_frames)} captured frame(s)?"
        )

        if confirm:
            self.captured_frames.clear()
            self._set_status("All frames cleared", "#00ff00")

    def _create_report(self):
        if not self.captured_frames:
            self._set_status("Error: No frames captured", "#ff0000")
            return

        # Ask user for report name
        report_name = tk.simpledialog.askstring(
            "Save Report",
            "Enter report name:",
            initialvalue=f"report_{len(self.captured_frames)}_frames"
        )

        if not report_name:  # User cancelled
            self._set_status("Report save cancelled", "#ff8800")
            return

        try:
            out_dir = reporting.save_report(
                self.captured_frames, report_name=report_name)
            self.last_report_dir = out_dir
            self._set_status(
                f"Report saved to {Path(out_dir).name}", "#00ff00")
        except Exception as e:
            self._set_status(f"Report error: {str(e)[:50]}", "#ff0000")

    def _save_report(self):
        if hasattr(self, 'last_report_dir'):
            self._set_status(
                f"Report at {Path(self.last_report_dir).name}", "#00ff00")
        else:
            self._set_status("No report created yet", "#ff8800")

    def _show_help(self):
        help_text = """Image Denoising UI - Help

Median: size = neighborhood size for median filter (good for salt-and-pepper noise).

Gaussian: sigma = standard deviation for Gaussian smoothing (good for AWGN).

Fourier: cutoff_fraction = fraction of max radius to keep; filter_type = ideal/gaussian/raised_cosine.

Wavelet: wavelet = basis (db4/db8/sym8/coif4/bior4.4); method = bayes (BayesShrink) or visu (VisuShrink).
  - Post-filter blending: helps improve PSNR while preserving SSIM.
  - Balance: weight for SSIM vs PSNR in Auto-tune (default 0.6 for SSIM, 0.4 for PSNR).

Controls:
  - Run: denoise with current params.
  - Auto-tune: search small grid around current params for best balanced score.
  - Capture frame: save current result to captured list.
  - Create Report: generate a report from all captured frames (saved to results/reports/).
  - Save Report: confirm report location.
"""
        messagebox.showinfo("Help", help_text)

    def _process_queue(self):
        try:
            while True:
                msg_type, data = self.queue.get_nowait()
                if msg_type == "run_done":
                    denoised, metrics, elapsed, params, alg = data
                    self.last_run = {
                        "denoised": denoised,
                        "metrics": metrics,
                        "elapsed": elapsed,
                        "params": params,
                        "alg": alg,
                        "name": self.current_item.get("name")
                    }
                    self._update_images()
                    psnr = metrics.get("psnr", 0)
                    ssim = metrics.get("ssim", 0)
                    self._set_status(
                        f"Complete! PSNR: {psnr:.2f}dB, SSIM: {ssim:.3f}", "#00ff00")
                    self.cancel_btn.config(state=tk.DISABLED)
                elif msg_type == "autotune_done":
                    alg, top_results = data
                    self._show_autotune_results(alg, top_results)
                    self.cancel_btn.config(state=tk.DISABLED)
                elif msg_type == "progress":
                    self._set_status(data, "#ff8800")
                elif msg_type == "cancelled":
                    self._set_status("Operation cancelled", "#ff8800")
                    self.cancel_btn.config(state=tk.DISABLED)
                elif msg_type == "error":
                    self._set_status(f"Error: {str(data)[:60]}", "#ff0000")
                    self.cancel_btn.config(state=tk.DISABLED)
        except queue.Empty:
            pass
        self.root.after(100, self._process_queue)

    def _show_autotune_results(self, alg, results):
        """Show top auto-tune results in a selection dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Auto-tune Results - {alg.upper()}")
        dialog.geometry("800x500")
        dialog.configure(bg="#2e2e2e")

        tk.Label(dialog, text=f"Top {len(results)} Results (click to select)",
                 bg="#2e2e2e", fg="white", font=("Arial", 12, "bold")).pack(pady=10)

        # Create frame with scrollbar for results
        container = tk.Frame(dialog, bg="#2e2e2e")
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        canvas = tk.Canvas(container, bg="#1e1e1e")
        scrollbar = tk.Scrollbar(
            container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#1e1e1e")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Add results as buttons
        for idx, result in enumerate(results):
            rank = idx + 1
            psnr = result["psnr"]
            ssim = result["ssim"]
            score = result["score"]
            params_str = ", ".join(
                [f"{k}={v}" for k, v in result["params"].items()])

            btn_text = f"#{rank}: PSNR={psnr:.2f}dB | SSIM={ssim:.3f} | Score={score:.3f}\n{params_str}"

            btn = tk.Button(scrollable_frame, text=btn_text, bg="#0078d4", fg="white",
                            font=("Arial", 9), anchor='w', justify=tk.LEFT,
                            command=lambda r=result: self._select_autotune_result(r, dialog))
            btn.pack(fill=tk.X, padx=10, pady=5)

        # Close button
        tk.Button(dialog, text="Cancel", command=dialog.destroy, bg="#d9534f", fg="white",
                  font=("Arial", 10, "bold")).pack(pady=10)

        self._set_status(
            f"Auto-tune complete! Select from {len(results)} results", "#00ff00")

    def _select_autotune_result(self, result, dialog):
        """Apply selected auto-tune result."""
        from scripts.utils import compute_metrics
        metrics = compute_metrics(
            self.current_item.get("clean"), result["image"])

        self.last_run = {
            "denoised": result["image"],
            "metrics": metrics,
            "elapsed": result["elapsed"],
            "params": result["params"],
            "alg": self.alg_var.get(),
            "name": self.current_item.get("name")
        }
        self._update_images()
        self._set_status(
            f"Applied: PSNR={result['psnr']:.2f}dB, SSIM={result['ssim']:.3f}", "#00ff00")
        dialog.destroy()


def main():
    root = tk.Tk()
    app = DenoisingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
