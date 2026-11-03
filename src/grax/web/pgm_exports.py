"""PGM run downloads: map values as CSV and the maps as a PNG figure."""

from __future__ import annotations

import csv
import io

import numpy as np

from .pgm_dashboard import MAP_KEYS

CSV_COLUMNS = (
    "energy_eV",
    "cff",
    "geometric_throughput_percent",
    "mirror_reflectivity_percent",
    "grating_efficiency_percent",
    "mirror_x_grating_percent",
    "total_throughput_percent",
)


def maps_csv(maps: dict[str, np.ndarray]) -> bytes:
    """Return one CSV row per energy and CFF point."""
    text = io.StringIO()
    writer = csv.writer(text)
    writer.writerow(CSV_COLUMNS)
    for ci, cff in enumerate(maps["cff"]):
        for ei, energy in enumerate(maps["energies"]):
            writer.writerow((energy, cff, *(maps[key][ci, ei] for key in MAP_KEYS)))
    return text.getvalue().encode("utf-8")


def maps_png(maps: dict[str, np.ndarray]) -> io.BytesIO:
    """Return the five maps as one PNG figure, ready to read."""
    from matplotlib.colors import PowerNorm
    from matplotlib.figure import Figure

    figure = Figure(figsize=(14, 12), layout="constrained")
    axes = figure.subplots(3, 2)
    axes[0, 0].axis("off")
    panels = {
        "geometry": (axes[0, 1], "Geometric throughput"),
        "mirror": (axes[1, 0], "Mirror reflectivity"),
        "grating": (axes[1, 1], "Grating efficiency"),
        "optical": (axes[2, 0], "Mirror × grating"),
        "total": (axes[2, 1], "Mirror × grating × geometry"),
    }
    for key, (axis, title) in panels.items():
        values = np.clip(maps[key], 0.0, 100.0)
        top = 100.0 if key == "geometry" else max(float(np.nanmax(values)), 1e-6)
        image = axis.pcolormesh(
            maps["energies"],
            maps["cff"],
            values,
            shading="auto",
            cmap="turbo",
            norm=PowerNorm(gamma=0.5, vmin=0.0, vmax=top),
        )
        axis.set_yscale("log")
        axis.set(title=title, xlabel="Photon energy E [eV]", ylabel="CFF")
        figure.colorbar(image, ax=axis, label="%")
    output = io.BytesIO()
    figure.savefig(output, format="png", dpi=150)
    output.seek(0)
    return output
