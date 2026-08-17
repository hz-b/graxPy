"""Generate the multi-condition measurement set the joint fit is run against.

The repository ships one measured curve, at a single grazing angle, so there is
no real dataset that spans several measurement conditions. This step stands in
for the experiment: it simulates the grating defined by the ``true_*`` values in
``example_config`` under each of the four conditions and adds Gaussian noise.

Always runs with ``solver="rcwa"``. The generated files are the fixed target the
fit is scored against, so they must not change when the fit is run with a
different solver -- recovering the same parameters from ``--solver neviere`` is
part of what this example demonstrates.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pandas as pd
from example_config import (
    diffraction_order,
    evaluation_energies_ev,
    fourier_orders,
    grazing_angle_deg,
    layer_thickness_nm,
    measurement_conditions,
    measurement_path,
    measurements_dir,
    noise_floor_path,
    noise_relative,
    noise_seed,
    optical_constants_dir,
    period_lpermm,
    polarization,
    simulation_backend,
    true_depth_nm,
    true_top_cap_thickness_nm,
    true_wall_angle_deg,
    true_width_to_period_ratio,
    x_resolution_nm,
    z_resolution_nm,
)

from grax import BatchSimulationRunner, LaminarGrating, monochromator_grazing_angles_deg


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


def build_true_grating() -> LaminarGrating:
    """Build the grating the synthetic measurements are generated from.

    Returns:
        The laminar grating at the known ``true_*`` parameter values.
    """

    return LaminarGrating(
        period_lpermm=period_lpermm,
        width_to_period_ratio=true_width_to_period_ratio,
        depth_nm=true_depth_nm,
        left_wall_angle_deg=true_wall_angle_deg,
        right_wall_angle_deg=true_wall_angle_deg,
        substrate_material=_load_material("n_Si_cxro.txt", "Si"),
        layer_material=_load_material("n_Pt_cxro.txt", "Pt"),
        layer_thickness_nm=layer_thickness_nm,
        top_cap_material=_load_material("n_C_cxro.txt", "C"),
        top_cap_thickness_nm=true_top_cap_thickness_nm,
        z_resolution_nm=z_resolution_nm,
        x_resolution_nm=x_resolution_nm,
    )


def resolved_condition(condition: dict[str, object]) -> dict[str, object]:
    """Fill one condition in with the run-level defaults it does not override.

    Args:
        condition: Entry from ``measurement_conditions``.

    Returns:
        The condition with every field resolved, mirroring what
        ``prepare_joint_measurements`` does inside the optimizer.
    """

    return {
        "angle_mode": condition.get("angle_mode", "fixed"),
        "grazing_angle_deg": condition.get("grazing_angle_deg", grazing_angle_deg),
        "cff": condition.get("cff"),
        "diffraction_order": int(condition.get("diffraction_order", diffraction_order)),
        "polarization": condition.get("polarization", polarization),
    }


def main() -> None:
    """Simulate and write one noisy measurement file per condition."""

    measurements_dir.mkdir(parents=True, exist_ok=True)
    grating = build_true_grating()
    energies = np.asarray(evaluation_energies_ev, dtype=float)
    generator = np.random.default_rng(noise_seed)

    cases: list[dict[str, object]] = []
    slots: list[tuple[str, int]] = []
    for condition in measurement_conditions:
        label = str(condition["label"])
        resolved = resolved_condition(condition)
        if resolved["angle_mode"] == "cff":
            angles = monochromator_grazing_angles_deg(
                energies,
                period_lpermm=period_lpermm,
                diffraction_order=int(resolved["diffraction_order"]),
                cff=float(resolved["cff"]),
            )
        else:
            angles = np.full(energies.shape, float(resolved["grazing_angle_deg"]), dtype=float)
        for point_index, (energy_ev, angle_deg) in enumerate(zip(energies, angles, strict=True)):
            cases.append(
                {
                    "case_id": f"{label}_{point_index}",
                    "grating": grating,
                    "energy_ev": float(energy_ev),
                    "grazing_angle_deg": float(angle_deg),
                    "diffraction_order": int(resolved["diffraction_order"]),
                    "polarization": str(resolved["polarization"]),
                    "fourier_orders": fourier_orders,
                }
            )
            slots.append((label, point_index))

    runner = BatchSimulationRunner(
        fourier_orders=fourier_orders,
        backend=simulation_backend,
        solver="rcwa",
        on_error="fail_fast",
    )
    clean: dict[str, np.ndarray] = {
        str(condition["label"]): np.full(energies.size, np.nan, dtype=float)
        for condition in measurement_conditions
    }
    for result in runner.run_cases(cases):
        label, point_index = slots[int(result.index)]
        clean[label][point_index] = float(result.selected_efficiency)

    squared_sigmas: list[float] = []
    for condition in measurement_conditions:
        label = str(condition["label"])
        efficiencies = clean[label]
        if not np.all(np.isfinite(efficiencies)):
            raise RuntimeError(f"Simulation did not return every point for {label!r}.")
        sigmas = noise_relative * np.abs(efficiencies)
        noisy = efficiencies + generator.normal(0.0, sigmas)
        squared_sigmas.extend((sigmas**2).tolist())
        output_path = measurement_path(label)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter=" ")
            for energy_ev, efficiency in zip(energies, noisy, strict=True):
                writer.writerow([f"{energy_ev:.4f}", f"{efficiency:.8f}"])
        print(
            f"{label}: {condition['description']} -> {output_path} "
            f"({efficiencies.size} points, mean efficiency {efficiencies.mean():.5f}, "
            f"mean sigma {sigmas.mean():.2e})"
        )

    # The best a perfect fit could do is reproduce the clean curves, leaving the
    # noise as residual. Recording that floor lets step 3 say whether a fit has
    # converged or merely run out of trials, without hardcoding a number.
    noise_floor = float(np.mean(squared_sigmas))
    noise_floor_path.write_text(
        json.dumps(
            {
                "noise_relative": noise_relative,
                "noise_seed": noise_seed,
                "expected_pooled_loss_floor": noise_floor,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Relative noise: {noise_relative:.1%} of each point (seed {noise_seed})")
    print(f"Expected pooled-loss floor: {noise_floor:.3g} -> {noise_floor_path}")
    print("True parameters the fit should recover:")
    print(f"  width_to_period_ratio = {true_width_to_period_ratio}")
    print(f"  depth_nm              = {true_depth_nm}")
    print(f"  wall_angle_deg        = {true_wall_angle_deg}")
    print(f"  top_cap_thickness_nm  = {true_top_cap_thickness_nm}")


if __name__ == "__main__":
    main()
