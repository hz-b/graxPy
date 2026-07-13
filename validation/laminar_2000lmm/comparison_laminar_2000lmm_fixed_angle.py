"""Compare measurement data with fixed-angle 2000 l/mm simulations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


example_root = Path(__file__).resolve().parent
results_dir = example_root / "results"
measurement_dir = example_root / "simulation"
output_file = results_dir / "comparison_laminar_2000lmm_fixed_angle.png"
comparison_order = -1

CASE_CONFIG = {
    1: {
        "measurement": measurement_dir / "lG2000-DLS-B07_ascan-(twt-non)_energy_1order_alpha-1deg.dat",
        "baseline": results_dir / "laminar_2000lmm_fixed_angle_alpha1deg_all_orders.csv",
        "layered": results_dir / "laminar_2000lmm_fixed_angle_alpha1deg_layered_all_orders.csv",
    },
    2: {
        "measurement": measurement_dir / "lG2000-DLS-B07_ascan-(twt-non)_energy_1order_alpha-2deg.dat",
        "baseline": results_dir / "laminar_2000lmm_fixed_angle_alpha2deg_all_orders.csv",
        "layered": results_dir / "laminar_2000lmm_fixed_angle_alpha2deg_layered_all_orders.csv",
    },
    4: {
        "measurement": measurement_dir / "lG2000-DLS-B07_ascan-(twt-non)_energy_1order_alpha-4deg.dat",
        "baseline": results_dir / "laminar_2000lmm_fixed_angle_alpha4deg_all_orders.csv",
        "layered": results_dir / "laminar_2000lmm_fixed_angle_alpha4deg_layered_all_orders.csv",
    },
}


def _load_order_curve(csv_path: Path) -> pd.DataFrame:
    """Return the selected diffraction-order curve from one all-orders CSV."""

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing results file: {csv_path}")

    df = pd.read_csv(csv_path)
    if "order" not in df.columns or "energy_ev" not in df.columns or "efficiency" not in df.columns:
        raise ValueError(f"Unexpected results schema in {csv_path}")
    curve = df[df["order"] == comparison_order].copy().sort_values("energy_ev")
    if curve.empty:
        raise ValueError(f"No order {comparison_order} data found in {csv_path}")
    return curve


def _load_measurement_curve(data_path: Path) -> pd.DataFrame:
    """Return one measurement curve with energy and efficiency columns."""

    if not data_path.exists():
        raise FileNotFoundError(f"Missing measurement file: {data_path}")

    return pd.read_csv(
        data_path,
        sep=r"\s+",
        engine="python",
        header=None,
        names=["energy_ev", "efficiency"],
    )


missing_results = []
for config in CASE_CONFIG.values():
    for csv_path in config.values():
        if not csv_path.exists():
            missing_results.append(str(csv_path))

if missing_results:
    raise FileNotFoundError(
        "Missing measurement or simulation file(s). Run the fixed-angle simulations first and confirm the measurement files are present:\n"
        + "\n".join(missing_results)
    )

figure, axes = plt.subplots(len(CASE_CONFIG), 1, figsize=(11, 14), sharex=True)

for axis, (angle_deg, config) in zip(axes, CASE_CONFIG.items(), strict=True):
    measurement_curve = _load_measurement_curve(config["measurement"])
    baseline_curve = _load_order_curve(config["baseline"])
    layered_curve = _load_order_curve(config["layered"])

    axis.plot(
        measurement_curve["energy_ev"],
        measurement_curve["efficiency"],
        color="black",
        linewidth=1.2,
        marker="o",
        markersize=2.2,
        label="Measurement",
    )
    axis.plot(
        baseline_curve["energy_ev"],
        baseline_curve["efficiency"],
        linewidth=1.8,
        label="Simulation without top layers",
    )
    axis.plot(
        layered_curve["energy_ev"],
        layered_curve["efficiency"],
        linestyle="--",
        linewidth=1.8,
        label="Simulation with top layers",
    )
    axis.set_title(f"Laminar 2000 l/mm Fixed-Angle Comparison at Alpha = {angle_deg} deg")
    axis.set_ylabel("Efficiency")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")

axes[-1].set_xlabel("Energy (eV)")

figure.tight_layout()
figure.savefig(output_file, dpi=200, bbox_inches="tight")
plt.close(figure)

print(f"Combined comparison plot saved to: {output_file}")
