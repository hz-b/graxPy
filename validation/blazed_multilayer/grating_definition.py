"""Grating and sweep definition for the blazed 2400 l/mm multilayer case.

Both `run_rcwa.py` and `run_neviere.py` import everything from here, so the two
solver runs are guaranteed to see the same grating, the same energy-angle grid
and the same truncation. Anything defined per runner instead could drift between
them, and the resulting comparison plot would show a "solver disagreement" that
is really a mismatched sweep.

The sweep grid is not a plain energy range: each point is an (energy, grazing
angle) pair taken from the DiffractMod reference table, so grax is evaluated at
exactly the geometry the reference code used.

Nothing in this module depends on which solver is used.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

import grax

CASE_ROOT = Path(__file__).resolve().parent
OPTICAL_CONSTANTS_DIR = CASE_ROOT / "optical_constants"
SIMULATION_DIR = CASE_ROOT / "simulation"
RESULTS_DIR = CASE_ROOT / "results"
REFERENCE_FILE = SIMULATION_DIR / "DiffractMod_CrC_d4.8_N60.dat"

# Grating and multilayer geometry.
PERIOD_LPERMM = 2400
BLAZE_ANGLE_DEG = 1.37
ANTI_BLAZE_ANGLE_DEG = 3.25
BILAYER_PERIOD_NM = 4.8
GAMMA = 0.4
N_BILAYERS = 60

# Sweep settings shared by both solvers.
POLARIZATION = "p"
DIFFRACTION_ORDER = 2
FOURIER_ORDERS = 35
X_RESOLUTION_NM = 0.1
Z_RESOLUTION_NM = 0.01

QUICK_FOURIER_ORDERS = 10
QUICK_X_RESOLUTION_NM = 1.0
QUICK_Z_RESOLUTION_NM = 1.0
# A quick run walks the reference table in coarse jumps instead of every row.
QUICK_REFERENCE_STEP = 100


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


def load_reference_table() -> pd.DataFrame:
    """Return the DiffractMod reference table driving the sweep grid.

    Returns:
        Energy, efficiency and grazing angle columns, numeric and NaN-free.
    """

    table = pd.read_csv(REFERENCE_FILE, sep=r"\s+", engine="python")
    table = table[["Energy", "Efficiency(GR)", "alpha"]].copy()
    table = table.apply(pd.to_numeric, errors="coerce").dropna()
    return table.reset_index(drop=True)


def build_reference_grid(*, quick: bool = False, stride: int = 1) -> pd.DataFrame:
    """Return the subsampled reference rows this sweep evaluates.

    Args:
        quick: Take coarse jumps through the reference table.
        stride: Keep every Nth reference row. Must be >= 1.

    Returns:
        Subsampled reference table.
    """

    if stride < 1:
        raise ValueError("stride must be >= 1.")
    step = (QUICK_REFERENCE_STEP * stride) if quick else stride
    return load_reference_table().iloc[::step].copy()


def build_grating(*, quick: bool = False) -> grax.BlazedGrating:
    """Return the blazed grating on its Cr/C multilayer stack.

    Args:
        quick: Use coarse resolutions for a fast smoke run.

    Returns:
        Configured blazed grating.
    """

    return grax.BlazedGrating(
        period_lpermm=PERIOD_LPERMM,
        blaze_angle_deg=BLAZE_ANGLE_DEG,
        anti_blaze_angle_deg=ANTI_BLAZE_ANGLE_DEG,
        coating_stack=build_multilayer_stack(),
        x_resolution_nm=QUICK_X_RESOLUTION_NM if quick else X_RESOLUTION_NM,
        z_resolution_nm=QUICK_Z_RESOLUTION_NM if quick else Z_RESOLUTION_NM,
    )


def build_multilayer_stack() -> grax.MultilayerStack:
    """Return the Cr/C multilayer stack under the grating profile.

    Returns:
        Configured multilayer stack.
    """

    return grax.MultilayerStack(
        substrate_material=load_optical_constants("Si"),
        material_a=load_optical_constants("Cr"),
        material_b=load_optical_constants("C"),
        d_period_nm=BILAYER_PERIOD_NM,
        gamma=GAMMA,
        n_bilayers=N_BILAYERS,
        top_material=load_optical_constants("C"),
    )


def build_cases(*, quick: bool = False, stride: int = 1):
    """Return the batch cases for this sweep, identical for either solver.

    Args:
        quick: Use the coarse smoke configuration.
        stride: Keep every Nth reference row.

    Returns:
        Iterable of case dictionaries.
    """

    reference = build_reference_grid(quick=quick, stride=stride)
    energy_angle_pairs = list(
        zip(
            reference["Energy"].to_numpy(dtype=float),
            reference["alpha"].to_numpy(dtype=float),
        )
    )
    return grax.energy_angle_cases(
        grating=build_grating(quick=quick),
        energy_angle_pairs=energy_angle_pairs,
        polarization=POLARIZATION,
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
        "all_orders_csv": RESULTS_DIR / f"blazed_multilayer_all_orders_{solver}.csv",
        "selected_order_plot": RESULTS_DIR / f"blazed_multilayer_order_2_{solver}.png",
        "profile_plot": RESULTS_DIR / "blazed_multilayer_profile.png",
        "stack_plot": RESULTS_DIR / "multilayer_stack_schematic.png",
        "checkpoint_dir": RESULTS_DIR / f"checkpoints_{solver}",
    }
