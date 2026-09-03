"""Shared baseline configuration for the joint measurement-fit example workflow.

The joint optimizer fits one parameter set against several measured curves at
once, and those curves do not have to differ only by grazing angle. Every
measurement below overrides a different condition, so the four of them together
exercise each axis the spec supports: angle, diffraction order, angle mode, and
polarization.

No multi-condition measurement set ships with the repository, so step 0
simulates these curves from the ``true_*`` parameters and adds noise. That makes
the example self-checking: step 3 compares the fitted values against the ones
the data was generated from.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

example_root = Path(__file__).resolve().parent
optical_constants_dir = example_root.parents[1] / "optical_constants"
measurements_dir = example_root / "measurements"
results_root = example_root / "results"

# Grating geometry held fixed during the fit.
period_lpermm = 400.0
layer_thickness_nm = 28.77
x_resolution_nm = 0.5
z_resolution_nm = 0.5

# The parameters step 0 generates the measurements from, and step 3 checks the
# fit against. Each sits inside its search bound below but away from the centre,
# so recovering them is not an artifact of the bounds.
true_width_to_period_ratio = 0.643
true_depth_nm = 14.62
true_wall_angle_deg = 12.4  # applied to both walls
true_top_cap_thickness_nm = 0.85

parameter_bounds = {
    "width_to_period_ratio": (0.55, 0.75),
    "depth_nm": (13.9, 15.9),
    "left_wall_angle_deg": (8.0, 18.0),
    "right_wall_angle_deg": (8.0, 18.0),
    "top_cap_thickness_nm": (0.3, 2.0),
}

# The right wall follows the left instead of being fitted separately, which keeps
# the search space small enough that a short example run converges. This is the
# same tied-wall idea as the laminar example's 0b step.
equality_constraints = {"right_wall_angle_deg": "left_wall_angle_deg"}

# Run-level measurement conditions. Each measurement inherits these unless it
# sets its own.
angle_mode = "fixed"
grazing_angle_deg = 4.0
diffraction_order = 1
polarization = "s"

fourier_orders = 12
evaluation_energies_ev = np.arange(120.0, 481.0, 24.0)

# The four measurement conditions, each differing from the run-level defaults in
# exactly one way so the plot in step 3 shows what each axis does.
measurement_conditions = [
    {
        "label": "alpha4deg_order1",
        "description": "run-level defaults: fixed 4 deg, order 1, s",
    },
    {
        "label": "alpha2deg_order1",
        "description": "different grazing angle",
        "grazing_angle_deg": 2.0,
    },
    {
        "label": "alpha4deg_order2",
        "description": "different diffraction order",
        "diffraction_order": 2,
    },
    {
        "label": "cff2p25_order1_p",
        "description": "different angle mode and polarization",
        "angle_mode": "cff",
        "cff": 2.25,
        "polarization": "p",
    },
]

# Measurement synthesis (step 0). The noise is relative to each point rather than
# absolute, because the order-2 curve is an order of magnitude weaker than the
# others and one absolute sigma would bury it while barely touching the rest.
noise_relative = 0.01
noise_seed = 11
noise_floor_path = measurements_dir / "noise_floor.json"

# Optimizer settings. The first pass stops at first_pass_trials and writes a
# checkpoint; step 2 resumes and extends the same run to total_trials.
#
# total_trials is cumulative across resumed runs, so raising it and rerunning
# step 2 asks only for the difference. Ax's model-fitting cost grows with the
# trial count, so the later trials are noticeably slower than the early ones.
first_pass_trials = 12
total_trials = 20
batch_size = 1
random_seed = 7
checkpoint_interval = 1
joint_loss_reduction = "pooled"
optimizer_backend = "auto"
simulation_backend = "numba"
optimizer_max_workers = "auto"


def measurement_path(label: str) -> Path:
    """Return the generated measurement file for one condition label.

    Args:
        label: Condition label from ``measurement_conditions``.

    Returns:
        Path to the two-column measurement file for that condition.
    """

    return measurements_dir / f"measured_{label}.csv"


def results_dir(solver: str) -> Path:
    """Return the results directory for one solver.

    Args:
        solver: Electromagnetic solver the fit ran with.

    Returns:
        Solver-suffixed results directory, so an rcwa and a neviere fit sit side
        by side instead of overwriting each other.
    """

    return results_root / f"joint_fit_{solver}"


def build_measurement_specs() -> list[dict[str, object]]:
    """Return the joint measurement specs for the fit.

    Returns:
        One spec mapping per condition, ready to pass as ``measurements``. Each
        spec carries only the conditions that differ from the run-level
        defaults; the rest are inherited.
    """

    specs: list[dict[str, object]] = []
    for condition in measurement_conditions:
        spec: dict[str, object] = {
            "label": condition["label"],
            "measurement_path": measurement_path(str(condition["label"])),
            "evaluation_energies_ev": list(evaluation_energies_ev),
        }
        for key in ("angle_mode", "grazing_angle_deg", "cff", "diffraction_order", "polarization"):
            if key in condition:
                spec[key] = condition[key]
        specs.append(spec)
    return specs
