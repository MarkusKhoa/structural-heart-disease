#!/usr/bin/env python3
"""
Visualize a 12-lead ECG from EchoNext .npy waveform files.

Expected default shape: (N, 1, 2500, 12) – 10s at 250 Hz, 12 leads.
This script handles shapes: (N,1,T,12), (N,T,12), or (T,12).

Usage examples:
  python visualize_ecg.py --waveforms ./echonext_dataset/EchoNext_train_waveforms.npy --index 0
  python visualize_ecg.py --waveforms ./echonext_dataset/EchoNext_test_waveforms.npy --index 42 --save ecg_42.png
  python visualize_ecg.py --waveforms ./echonext_dataset/EchoNext_val_waveforms.npy --index 10 --lead-names I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator


DEFAULT_LEAD_NAMES = [
    "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"
]


def parse_args():
    p = argparse.ArgumentParser(description="Visualize a 12-lead ECG sample from an EchoNext .npy file")
    p.add_argument("--waveforms", type=str, required=True, help="Path to .npy waveforms (e.g., EchoNext_train_waveforms.npy)")
    p.add_argument("--index", type=int, default=0, help="Sample index to visualize (default: 0)")
    p.add_argument("--fs", type=float, default=250.0, help="Sampling rate in Hz (default: 250)")
    p.add_argument("--lead-names", type=str, default=",".join(DEFAULT_LEAD_NAMES), help="Comma-separated list of 12 lead names")
    p.add_argument("--mmap", action="store_true", help="Memory-map the .npy file instead of loading fully")
    p.add_argument("--save", type=str, default=None, help="Optional path to save figure (e.g., output.png)")
    p.add_argument("--dpi", type=int, default=200, help="Figure DPI when saving (default: 200)")
    p.add_argument("--width", type=float, default=12.0, help="Figure width in inches (default: 12)")
    p.add_argument("--height", type=float, default=8.0, help="Figure height in inches (default: 8)")
    p.add_argument("--no-show", action="store_true", help="Do not show the figure interactively")
    return p.parse_args()


def load_waveforms(path: str, mmap: bool = False) -> np.ndarray:
    mode = "r" if mmap else None
    arr = np.load(path, mmap_mode=mode)
    return arr


def extract_sample(waveforms: np.ndarray, index: int) -> np.ndarray:
    """Return a single sample as shape (T, 12)."""
    if waveforms.ndim == 4:
        # (N, 1, T, 12)
        sample = waveforms[index]
        if sample.shape[0] == 1:
            sample = sample[0]  # (T, 12)
        else:
            # Unexpected, but try to squeeze the first dim
            sample = np.squeeze(sample, axis=0)
        # Now sample should be (T, 12)
    elif waveforms.ndim == 3:
        # (N, T, 12)
        sample = waveforms[index]
    elif waveforms.ndim == 2:
        # (T, 12) already a single sample file
        if index != 0:
            raise IndexError("Index > 0 but provided array is a single-sample (T,12)")
        sample = waveforms
    else:
        raise ValueError(f"Unsupported waveforms ndim={waveforms.ndim}. Expected 2-4 dims.")

    # Ensure final shape (T, 12)
    if sample.ndim != 2 or sample.shape[1] != 12:
        raise ValueError(f"Sample shape must be (T,12). Got {sample.shape}")
    return np.asarray(sample)


def parse_lead_names(arg: str) -> List[str]:
    leads = [s.strip() for s in arg.split(",")]
    if len(leads) != 12:
        raise ValueError(f"--lead-names must specify exactly 12 names, got {len(leads)}")
    return leads


def plot_ecg(sample: np.ndarray, fs: float, lead_names: List[str], width: float, height: float):
    T = sample.shape[0]
    duration = T / fs
    t = np.arange(T) / fs

    fig, axes = plt.subplots(4, 3, figsize=(width, height), sharex=True)
    axes = axes.ravel()

    for i in range(12):
        ax = axes[i]
        ax.plot(t, sample[:, i], color="black", linewidth=0.8)
        ax.set_title(lead_names[i], fontsize=10)
        ax.set_xlim(0, duration)
        # ECG-style grid: 0.2 s major, 0.04 s minor on x-axis
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(MultipleLocator(0.04))
        # Y grid: choose reasonable major loc based on data range
        yrange = np.nanmax(sample[:, i]) - np.nanmin(sample[:, i])
        ymaj = 0.5 if yrange <= 5 else max(1.0, round(yrange / 8, 1))
        ax.yaxis.set_major_locator(MultipleLocator(ymaj))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(which="major", color="#e0e0e0")
        ax.grid(which="minor", color="#f5f5f5", linestyle=":")

        # Only label x-axis on bottom row
        if i < 9:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Time (s)")

    fig.suptitle("12-lead ECG", fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    return fig


def main():
    args = parse_args()

    waveforms = load_waveforms(args.waveforms, mmap=args.mmap)
    print(f"Loaded waveforms from {args.waveforms} with shape {waveforms.shape}")
    if args.index < 0 or args.index >= (waveforms.shape[0] if waveforms.ndim >= 3 else 1):
        raise IndexError(f"Index {args.index} out of bounds for waveforms with shape {waveforms.shape}")

    sample = extract_sample(waveforms, args.index)
    print(f"Visualizing sample index {args.index} with shape {sample.shape}")
    lead_names = parse_lead_names(args.lead_names)

    fig = plot_ecg(sample, fs=args.fs, lead_names=lead_names, width=args.width, height=args.height)

    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved figure to {out_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
