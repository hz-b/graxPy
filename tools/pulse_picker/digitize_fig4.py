"""Digitize the 0 V and 9 V traces from the left panel of paper Fig. 4.

The pixel calibration and target colours are recorded here so the derived CSV
can be regenerated and reviewed. The output is an estimate from a raster
figure, never a substitute for the experiment's raw data.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

TRACES = {"0": (0, 0, 250), "9": (176, 237, 148)}
X_ZERO_PX = 370.0
PX_PER_DEG = (648.0 - 93.0) / 0.04
Y_ZERO_PX = 436.0
Y_ONE_PX = 23.0


def extract_trace(image: np.ndarray, rgb: tuple[int, int, int]) -> np.ndarray:
    """Find each coloured trace's middle pixel per x column."""
    target = np.asarray(rgb, dtype=float)
    out = np.full(image.shape[1], np.nan)
    for x in range(96, 645):
        # The legend crosses the upper-right of the panel. Away from the
        # Bragg peak, the plotted traces remain in the lower half.
        distance = abs(x - X_ZERO_PX)
        y_start = 375 if distance > 185 else (250 if distance > 100 else 22)
        colours = image[y_start:437, x, :].astype(float)
        distances = np.linalg.norm(colours - target, axis=1)
        candidates = np.flatnonzero(distances < 40.0)
        if candidates.size:
            out[x] = y_start + float(np.median(candidates))
    return out


def main() -> None:
    """Write uniformly sampled digitized points with pixel-level uncertainty."""
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    image = np.asarray(Image.open(args.image).convert("RGB"))
    traces = {voltage: extract_trace(image, rgb) for voltage, rgb in TRACES.items()}
    samples = np.arange(100, 641, 5)
    rows = []
    for voltage, pixels in traces.items():
        known = np.flatnonzero(np.isfinite(pixels))
        if known.size < 20:
            raise ValueError(f"Could not recover {voltage} V trace from figure")
        interpolated = np.interp(samples, known, pixels[known])
        for x, y in zip(samples, interpolated, strict=True):
            rows.append(
                {
                    "voltage_v": voltage,
                    "angle_offset_deg": f"{(x - X_ZERO_PX) / PX_PER_DEG:.8f}",
                    "normalized_intensity": (
                        f"{max(0.0, (Y_ZERO_PX - y) / (Y_ZERO_PX - Y_ONE_PX)):.6f}"
                    ),
                    "angle_uncertainty_deg": f"{2.0 / PX_PER_DEG:.8f}",
                    "intensity_uncertainty": f"{3.0 / (Y_ZERO_PX - Y_ONE_PX):.6f}",
                }
            )
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} digitized points to {args.csv}")


if __name__ == "__main__":
    main()
