"""Grating and sweep definition for the laminar 400 l/mm validation case.

Both `run_rcwa.py` and `run_neviere.py` import everything from here, so the two
solver runs are guaranteed to see the same grating, the same energy grid and the
same truncation. Anything defined per runner instead could drift between them,
and the resulting comparison plot would show a "solver disagreement" that is
really a mismatched sweep.

Nothing in this module depends on which solver is used.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import grax

CASE_ROOT = Path(__file__).resolve().parent
OPTICAL_CONSTANTS_DIR = CASE_ROOT / "optical_constants"
RESULTS_DIR = CASE_ROOT / "results"
MEASUREMENT_FILE = CASE_ROOT / "measured_alpha4deg_order1.csv"

# Sweep settings shared by both solvers.
GRAZING_ANGLE_DEG = 4.0
POLARIZATION = "p"
DIFFRACTION_ORDER = 1
FOURIER_ORDERS = 30
ROUGHNESS_SIGMA_NM = 0.5

QUICK_ENERGIES_EV = np.asarray([100.0, 300.0, 600.0], dtype=float)
QUICK_FOURIER_ORDERS = 5


@lru_cache(maxsize=None)
def load_optical_constants(name: str) -> pd.DataFrame:
    """Return one optical-constants table by material name.

    Cached so repeated calls return the *same* object. ``MultilayerStack``
    identifies ``top_material`` by matching it against ``material_a`` or
    ``material_b``, which fails if each call hands back a fresh DataFrame.

    Args:
        name: Material name matching an ``OC_<name>_SSTR.dat`` file.

    Returns:
        Optical-constants table tagged with the material name.
    """

    table = pd.read_csv(
        OPTICAL_CONSTANTS_DIR / f"OC_{name}_SSTR.dat",
        sep=r"\s*,\s*|\s+",
        engine="python",
    )
    table.attrs["name"] = name
    return table


def build_grating(*, quick: bool = False) -> grax.LaminarGrating:
    """Return the laminar grating for this validation case.

    Args:
        quick: Use coarse resolutions for a fast smoke run.

    Returns:
        Configured laminar grating.
    """

    return grax.LaminarGrating(
        period_lpermm=400,
        width_to_period_ratio=0.67,
        depth_nm=14.9,
        left_wall_angle_deg=15.0,
        right_wall_angle_deg=15.0,
        substrate_material=load_optical_constants("Si"),
        layer_material=load_optical_constants("Pt"),
        layer_thickness_nm=28.77,
        top_cap_material=load_optical_constants("C"),
        top_cap_thickness_nm=0.7,
        z_resolution_nm=2.0 if quick else 0.1,
        x_resolution_nm=10.0 if quick else 0.1,
    )


def build_energies_ev(*, quick: bool = False, stride: int = 1) -> np.ndarray:
    """Return the photon energies swept by this case.

    Args:
        quick: Use the three-point smoke grid.
        stride: Keep every Nth energy. Must be >= 1.

    Returns:
        Photon energies in electronvolts.
    """

    if stride < 1:
        raise ValueError("stride must be >= 1.")
    energies = QUICK_ENERGIES_EV if quick else np.arange(50.0, 650.1, 1.0)
    return energies[::stride]


def build_cases(*, quick: bool = False, stride: int = 1):
    """Return the batch cases for this sweep, identical for either solver.

    Args:
        quick: Use the coarse smoke configuration.
        stride: Keep every Nth energy.

    Returns:
        Iterable of case dictionaries.
    """

    grating = build_grating(quick=quick)
    cases = grax.fixed_angle_cases(
        grating=grating,
        energies_ev=build_energies_ev(quick=quick, stride=stride),
        grazing_angle_deg=GRAZING_ANGLE_DEG,
        polarization=POLARIZATION,
    )
    return (
        dict(case, label="fixed-angle", roughness_sigma_nm=ROUGHNESS_SIGMA_NM)
        for case in cases
    )


def output_paths(solver: str) -> dict[str, Path]:
    """Return the output paths for one solver's run.

    The checked-in artifacts under ``results/`` keep their historical unsuffixed
    names; every fresh run writes to a ``_rcwa`` or ``_neviere`` sibling.

    Args:
        solver: ``"rcwa"`` or ``"neviere"``.

    Returns:
        Mapping of output name to path.
    """

    if solver not in ("rcwa", "neviere"):
        raise ValueError(f"solver must be 'rcwa' or 'neviere', got {solver!r}.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "all_orders_csv": RESULTS_DIR / f"laminar_fixed_angle_all_orders_{solver}.csv",
        "measurement_plot": RESULTS_DIR / f"laminar_fixed_angle_comparison_{solver}.png",
        "profile_plot": RESULTS_DIR / "laminar_fixed_angle_profile.png",
    }
