"""Compare the joint fit against the parameters the measurements were built from.

Because step 0 generated the data from known values, this step can do something
a real fit cannot: report how close each fitted parameter came to the truth. That
is more informative than the loss alone, because a low loss does not by itself
mean every parameter was well constrained by the data.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from example_config import (  # noqa: E402
    measurement_conditions,
    noise_floor_path,
    results_dir,
    true_depth_nm,
    true_top_cap_thickness_nm,
    true_wall_angle_deg,
    true_width_to_period_ratio,
)
from joint_fit import solver_argument_parser  # noqa: E402

TRUE_PARAMETERS = {
    "width_to_period_ratio": true_width_to_period_ratio,
    "depth_nm": true_depth_nm,
    "left_wall_angle_deg": true_wall_angle_deg,
    "right_wall_angle_deg": true_wall_angle_deg,
    "top_cap_thickness_nm": true_top_cap_thickness_nm,
}


def main() -> None:
    """Plot every condition's fit and print the parameter-recovery table."""

    args = solver_argument_parser(__doc__ or "Plot a joint fit comparison").parse_args()
    output_dir = results_dir(args.solver)
    comparison_path = output_dir / "best_fit_comparison.csv"
    if not comparison_path.is_file():
        print(f"No fit found at {comparison_path}. Run 1_fit_joint.py first.")
        raise SystemExit(1)

    rows_by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    with comparison_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows_by_label[row["label"]].append(row)

    descriptions = {
        str(condition["label"]): str(condition["description"])
        for condition in measurement_conditions
    }

    labels = [str(condition["label"]) for condition in measurement_conditions]
    figure, axes = plt.subplots(len(labels), 1, figsize=(10, 3.4 * len(labels)), squeeze=False)
    for axis, label in zip(axes[:, 0], labels, strict=True):
        rows = rows_by_label[label]
        energies = [float(row["energy_ev"]) for row in rows]
        measured = [float(row["measured_efficiency"]) for row in rows]
        simulated = [float(row["simulated_efficiency"]) for row in rows]
        first = rows[0]
        geometry = (
            f"cff = {first['cff']}"
            if first["angle_mode"] == "cff"
            else f"alpha = {first['grazing_angle_deg']} deg"
        )

        axis.plot(energies, measured, "o", markersize=4, label="Generated measurement")
        axis.plot(energies, simulated, "-", linewidth=1.4, label="Joint best fit")
        axis.set_title(
            f"{label} -- {descriptions[label]}\n"
            f"{geometry}, order {first['diffraction_order']}, {first['polarization']}-pol"
        )
        axis.set_xlabel("Photon Energy (eV)")
        axis.set_ylabel("Diffraction Efficiency")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")

    figure.tight_layout()
    plot_path = output_dir / f"joint_fit_comparison_{args.solver}.png"
    figure.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    payload = json.loads((output_dir / "best_result.json").read_text(encoding="utf-8"))
    fitted = payload["best_grating_parameters"]

    print(f"Solver: {payload['solver']}")
    print(f"Completed trials: {payload['completed_trials']}")
    noise_floor = json.loads(noise_floor_path.read_text(encoding="utf-8"))
    floor = float(noise_floor["expected_pooled_loss_floor"])
    print(f"Joint loss ({payload['joint_loss_reduction']}): {payload['best_loss']:.6g}")
    print(
        f"Noise floor: {floor:.3g} "
        f"(mean sigma^2 of the {noise_floor['noise_relative']:.0%} noise step 0 added)"
    )
    print()
    print("Per-condition loss:")
    for label, loss in sorted(payload["per_measurement_best_losses"].items()):
        print(f"  {label:<20s} {loss:.6g}")
    print()
    print(f"{'parameter':<24s} {'true':>10s} {'fitted':>10s} {'error':>10s} {'rel':>8s}")
    recovery: list[tuple[str, float]] = []
    for name, true_value in TRUE_PARAMETERS.items():
        if name not in fitted:
            continue
        fitted_value = float(fitted[name])
        error = fitted_value - float(true_value)
        relative = error / float(true_value)
        recovery.append((name, abs(relative)))
        print(
            f"{name:<24s} {float(true_value):10.4f} {fitted_value:10.4f} "
            f"{error:10.4f} {relative:7.1%}"
        )

    tightest = min(recovery, key=lambda item: item[1])
    loosest = max(recovery, key=lambda item: item[1])
    at_noise_floor = float(payload["best_loss"]) <= 2.0 * floor

    print()
    print(
        f"Tightest: {tightest[0]} to {tightest[1]:.1%}. "
        f"Loosest: {loosest[0]} to {loosest[1]:.1%}."
    )
    if at_noise_floor:
        print(
            "The joint loss has reached the noise floor, so the residual parameter error is\n"
            "not something more trials can fix. Several parameter combinations reproduce\n"
            "these curves equally well within the noise, and the optimizer has no way to\n"
            "prefer the true one. Which parameters come back loosely varies between runs for\n"
            "the same reason -- Ax's model-based stage is not bit-reproducible from the seed\n"
            "alone. To pin a degenerate parameter down you need measurements that respond to\n"
            "it: more conditions, a wider energy range, or less noise, not a longer run."
        )
    else:
        print(
            "The joint loss is still above the noise floor, so the fit has not converged.\n"
            "Raise total_trials in example_config.py and rerun 2_resume_and_extend.py: it\n"
            "picks up from the checkpoint rather than starting over."
        )
    print(f"Comparison plot: {plot_path}")


if __name__ == "__main__":
    main()
