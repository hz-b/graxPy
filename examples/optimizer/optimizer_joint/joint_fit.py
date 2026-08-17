"""Shared joint-fit plumbing for steps 1 and 2.

Both steps run the same problem; they differ only in the trial budget and
whether they resume. Keeping the spec in one place means the resume in step 2 is
guaranteed to present the optimizer with the same problem fingerprint, which is
exactly what the checkpoint's resume guard checks for.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
from example_config import (
    angle_mode,
    batch_size,
    build_measurement_specs,
    checkpoint_interval,
    diffraction_order,
    equality_constraints,
    fourier_orders,
    grazing_angle_deg,
    joint_loss_reduction,
    layer_thickness_nm,
    optical_constants_dir,
    optimizer_backend,
    optimizer_max_workers,
    parameter_bounds,
    period_lpermm,
    polarization,
    random_seed,
    results_dir,
    x_resolution_nm,
    z_resolution_nm,
)

from grax import LaminarGrating


def solver_argument_parser(description: str) -> argparse.ArgumentParser:
    """Return the argument parser shared by the fit steps.

    Args:
        description: Help text for the step.

    Returns:
        A parser carrying the repository's standard ``--solver`` flag.
    """

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--solver",
        choices=("rcwa", "neviere"),
        default="rcwa",
        help="Electromagnetic solver to run. Both compute every diffraction order; "
        "they differ only in how each layer is crossed in z.",
    )
    return parser


def _load_material(filename: str, name: str) -> pd.DataFrame:
    """Load one optical-constants table.

    Args:
        filename: File name inside the shared optical-constants directory.
        name: Material name recorded on the table.

    Returns:
        The optical constants, tagged with the material name.
    """

    table = pd.read_csv(
        optical_constants_dir / filename,
        skiprows=1,
        sep=r"\s*,\s*|\s+",
        engine="python",
    )
    table.attrs["name"] = name
    return table


silicon = _load_material("n_Si_cxro.txt", "Si")
platinum = _load_material("n_Pt_cxro.txt", "Pt")
carbon = _load_material("n_C_cxro.txt", "C")


def build_grating(parameters: dict[str, float]) -> LaminarGrating:
    """Build the laminar grating from one trial's resolved parameters.

    Args:
        parameters: Resolved grating parameters for the trial, with the tied
            right wall already filled in from the left.

    Returns:
        The grating to simulate for this trial.
    """

    return LaminarGrating(
        period_lpermm=period_lpermm,
        width_to_period_ratio=float(parameters["width_to_period_ratio"]),
        depth_nm=float(parameters["depth_nm"]),
        left_wall_angle_deg=float(parameters["left_wall_angle_deg"]),
        right_wall_angle_deg=float(parameters["right_wall_angle_deg"]),
        substrate_material=silicon,
        layer_material=platinum,
        layer_thickness_nm=layer_thickness_nm,
        top_cap_material=carbon,
        top_cap_thickness_nm=float(parameters["top_cap_thickness_nm"]),
        z_resolution_nm=z_resolution_nm,
        x_resolution_nm=x_resolution_nm,
    )


def build_spec(*, solver: str, total_trials: int, resume: bool) -> dict[str, object]:
    """Assemble the joint measurement-fit spec.

    Args:
        solver: Electromagnetic solver to run every trial with.
        total_trials: Cumulative trial budget, counted across resumed runs.
        resume: Whether to continue from an existing checkpoint.

    Returns:
        The spec mapping passed to ``optimize_to_joint_measurements``.
    """

    return {
        "build_grating": build_grating,
        "parameter_bounds": dict(parameter_bounds),
        "equality_constraints": dict(equality_constraints),
        "output_dir": results_dir(solver),
        "measurements": build_measurement_specs(),
        "angle_mode": angle_mode,
        "grazing_angle_deg": grazing_angle_deg,
        "diffraction_order": diffraction_order,
        "polarization": polarization,
        "fourier_orders": fourier_orders,
        "solver": solver,
        "joint_loss_reduction": joint_loss_reduction,
        "experiment_name": "joint_fit",
        "total_trials": total_trials,
        "batch_size": batch_size,
        "random_seed": random_seed,
        "resume": resume,
        "checkpoint_interval": checkpoint_interval,
        "validate_physical_results": True,
        "save_best_fit_plot": True,
        "save_loss_plot": True,
        "save_comparison_csv": True,
        "backend": optimizer_backend,
        "max_workers": optimizer_max_workers,
    }


def report(result: object, *, solver: str, step: str) -> None:
    """Print the outcome of one fit step and save the fitted parameters.

    Args:
        result: Joint optimization result.
        solver: Solver the fit ran with.
        step: Human-readable step name for the heading.
    """

    output_dir = results_dir(solver)
    fitted_parameters_path = output_dir / "fitted_parameters.json"
    payload = json.loads(result.result_json_path.read_text(encoding="utf-8"))
    payload["result_json_path"] = str(result.result_json_path)
    payload["trial_history_csv_path"] = str(result.trial_history_csv_path)
    payload["comparison_csv_path"] = (
        None if result.comparison_csv_path is None else str(result.comparison_csv_path)
    )
    fitted_parameters_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"--- {step} (solver={solver}) ---")
    print(f"Completed trials: {result.completed_trials}")
    print(f"Best joint loss: {result.best_loss:.6g}")
    for label, loss in sorted(result.per_measurement_best_losses.items()):
        print(f"  {label}: {loss:.6g}")
    print(f"Best parameters: {result.best_parameters}")
    print(f"Fitted parameters JSON: {fitted_parameters_path}")
    print(f"Best result JSON: {result.result_json_path}")
    print(f"Trial history CSV: {result.trial_history_csv_path}")
    if result.best_fit_plot_path is not None:
        print(f"Best-fit plot: {result.best_fit_plot_path}")
    if result.loss_history_plot_path is not None:
        print(f"Loss-history plot: {result.loss_history_plot_path}")
    if result.comparison_csv_path is not None:
        print(f"Comparison CSV: {result.comparison_csv_path}")
