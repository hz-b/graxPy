"""Helper functions for the roughness-correlation-length comparison example."""

from __future__ import annotations

from pathlib import Path

import grax
from helpers_grating_plot import save_grating_plot as _save_grating_plot

RoughnessRun = tuple[str, float, float | None, int]


def correlation_slug(correlation_length_nm: float | None) -> str:
    """Return a filename-safe correlation-length identifier."""
    if correlation_length_nm is None:
        return "na"
    return str(correlation_length_nm).replace(".", "p")


def _supercell_suffix(num_supercells: int) -> str:
    """Return a filename suffix for the supercell count, empty when it's 1."""
    return "" if num_supercells == 1 else f"_supercells_{num_supercells}"


def run_label(
    roughness_kind: str,
    roughness_sigma_nm: float,
    correlation_length_nm: float | None,
    num_supercells: int = 1,
) -> str:
    """Return a readable label for one run."""
    if roughness_sigma_nm == 0.0:
        return "sigma zero"
    if roughness_kind == "debye-waller":
        return f"Debye-Waller sigma={roughness_sigma_nm:.1f} nm"
    label = f"random-interface sigma={roughness_sigma_nm:.1f} nm, correlation={correlation_length_nm:.0f} nm"
    if num_supercells != 1:
        label += f", supercells={num_supercells}"
    return label


def run_title(
    roughness_kind: str,
    roughness_sigma_nm: float,
    correlation_length_nm: float | None,
    num_supercells: int = 1,
) -> str:
    """Return a terminal title for one simulation run."""
    return f"Running {run_label(roughness_kind, roughness_sigma_nm, correlation_length_nm, num_supercells)}"


def csv_path(
    output_dir: Path,
    roughness_kind: str,
    roughness_sigma_nm: float,
    correlation_length_nm: float | None,
    num_supercells: int = 1,
) -> Path:
    """Return the CSV path for one run."""
    sigma_slug = str(roughness_sigma_nm).replace(".", "p")
    corr_slug = correlation_slug(correlation_length_nm)
    suffix = _supercell_suffix(num_supercells)
    return (
        output_dir
        / f"roughness_correlation_{roughness_kind}_sigma_{sigma_slug}_corr_{corr_slug}{suffix}_all_orders.csv"
    )


def grating_plot_path(
    output_dir: Path,
    roughness_kind: str,
    roughness_sigma_nm: float,
    correlation_length_nm: float | None,
    num_supercells: int = 1,
) -> Path:
    """Return the PDF path for one whole-grating geometry plot."""
    sigma_slug = str(roughness_sigma_nm).replace(".", "p")
    corr_slug = correlation_slug(correlation_length_nm)
    suffix = _supercell_suffix(num_supercells)
    return (
        output_dir
        / f"roughness_correlation_{roughness_kind}_sigma_{sigma_slug}_corr_{corr_slug}{suffix}_grating.pdf"
    )


def save_grating_plot(
    grating: grax.LaminarGrating,
    output_dir: Path,
    *,
    roughness_kind: str,
    roughness_sigma_nm: float,
    correlation_length_nm: float | None,
    num_supercells: int = 1,
) -> Path:
    """Save a whole-grating PDF of the generated material geometry."""
    output_path = grating_plot_path(
        output_dir, roughness_kind, roughness_sigma_nm, correlation_length_nm, num_supercells
    )
    title = run_label(roughness_kind, roughness_sigma_nm, correlation_length_nm, num_supercells)
    return _save_grating_plot(grating, output_path, title=title)
