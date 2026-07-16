"""Helper functions for the roughness-kind comparison example.

These functions carry no example configuration of their own; every tunable is
passed in from ``roughness_kind_comparison.py`` so all the knobs live in one place.
"""

from __future__ import annotations

from pathlib import Path

import grax
from grax.stacks import LayerSpec, assemble_custom_stack
from helpers_grating_plot import save_grating_plot as _save_grating_plot


RoughnessRun = tuple[str, float, int]


def roughness_slug(roughness_sigma_nm: float) -> str:
    """Return a filename-safe roughness identifier."""
    return str(roughness_sigma_nm).replace(".", "p")


def _supercell_suffix(num_supercells: int) -> str:
    """Return a filename suffix for the supercell count, empty when it's 1."""
    return "" if num_supercells == 1 else f"_supercells_{num_supercells}"


def case_label(roughness_kind: str, roughness_sigma_nm: float, num_supercells: int = 1) -> str:
    """Return a readable label for one roughness run."""
    if roughness_sigma_nm == 0.0:
        return "sigma zero"
    label = f"{roughness_kind} sigma={roughness_sigma_nm:.1f} nm"
    if num_supercells != 1:
        label += f", supercells={num_supercells}"
    return label


def run_title(roughness_kind: str, roughness_sigma_nm: float, num_supercells: int = 1) -> str:
    """Return a terminal title for one simulation run."""
    if roughness_sigma_nm == 0.0:
        return "Running sigma zero baseline with no roughness"
    return f"Running {case_label(roughness_kind, roughness_sigma_nm, num_supercells)}"


def csv_path(output_dir: Path, roughness_kind: str, roughness_sigma_nm: float, num_supercells: int = 1) -> Path:
    """Return the CSV path for one roughness run."""
    slug = roughness_slug(roughness_sigma_nm)
    suffix = _supercell_suffix(num_supercells)
    return output_dir / f"roughness_kind_comparison_{roughness_kind}_sigma_{slug}{suffix}_all_orders.csv"


def grating_plot_path(
    output_dir: Path, roughness_kind: str, roughness_sigma_nm: float, num_supercells: int = 1
) -> Path:
    """Return the PDF path for one whole-grating geometry plot."""
    slug = roughness_slug(roughness_sigma_nm)
    suffix = _supercell_suffix(num_supercells)
    return output_dir / f"roughness_kind_comparison_{roughness_kind}_sigma_{slug}{suffix}_grating.pdf"


def order_spectrum_plot_path(
    output_dir: Path, roughness_kind: str, roughness_sigma_nm: float, num_supercells: int = 1
) -> Path:
    """Return the PNG path for one order-spectrum-at-one-energy plot."""
    slug = roughness_slug(roughness_sigma_nm)
    suffix = _supercell_suffix(num_supercells)
    return output_dir / f"roughness_kind_comparison_{roughness_kind}_sigma_{slug}{suffix}_order_spectrum.png"


def build_grating(
    roughness_kind: str,
    roughness_sigma_nm: float,
    *,
    substrate_material: str,
    coating_layers_nm: list[tuple[str, float]],
    period_lpermm: float,
    width_to_period_ratio: float,
    depth_nm: float,
    wall_angle_deg: float,
    x_resolution_nm: float,
    z_resolution_nm: float,
    roughness_seed: int,
    roughness_correlation_length_nm: float | None = None,
    roughness_num_supercells: int = 1,
) -> grax.LaminarGrating:
    """Build the example grating with the selected per-layer roughness model."""
    # Assign the same roughness to the substrate boundary and to every coating
    # layer through the per-layer roughness system. The grating-level
    # ``RoughnessSpec`` only carries the kind (and seed); its ``sigma_nm`` stays
    # 0.0 so the per-interface values drive the roughness. ``substrate_roughness_sigma_nm``
    # roughens interface 0 (the substrate/coating boundary), which otherwise
    # falls back to the default and stays flat.
    stack = assemble_custom_stack(
        substrate_material=substrate_material,
        substrate_roughness_sigma_nm=roughness_sigma_nm,
        layers_bottom_up=[
            LayerSpec(
                material=material,
                thickness_nm=thickness_nm,
                roughness_sigma_nm=roughness_sigma_nm,
            )
            for material, thickness_nm in coating_layers_nm
        ],
    )
    roughness = None
    if roughness_sigma_nm > 0.0:
        roughness = grax.RoughnessSpec(
            kind=roughness_kind,
            sigma_nm=0.0,
            seed=roughness_seed,
            correlation_length_nm=roughness_correlation_length_nm,
            # num_supercells > 1 is only valid for random-interface roughness;
            # RoughnessSpec rejects it for debye-waller, so keep it at the
            # default there regardless of the caller's config.
            num_supercells=roughness_num_supercells if roughness_kind == "random-interface" else 1,
        )
    return grax.LaminarGrating(
        period_lpermm=period_lpermm,
        width_to_period_ratio=width_to_period_ratio,
        depth_nm=depth_nm,
        left_wall_angle_deg=wall_angle_deg,
        right_wall_angle_deg=wall_angle_deg,
        substrate_material=substrate_material,
        coating_stack=stack,
        x_resolution_nm=x_resolution_nm,
        z_resolution_nm=z_resolution_nm,
        roughness=roughness,
    )


def save_grating_plot(
    grating: grax.LaminarGrating,
    output_dir: Path,
    *,
    roughness_kind: str,
    roughness_sigma_nm: float,
    num_supercells: int = 1,
) -> Path:
    """Save a whole-grating PDF of the generated material geometry."""
    output_path = grating_plot_path(output_dir, roughness_kind, roughness_sigma_nm, num_supercells)
    title = case_label(roughness_kind, roughness_sigma_nm, num_supercells)
    return _save_grating_plot(grating, output_path, title=title)
