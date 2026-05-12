"""Batch simulation example with manual user-defined depth-sweep cases."""

from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import grax as rp
from xrt.backends.raycing import materials as xrt_materials

silicon = xrt_materials.Material("Si", rho=2.33, table="Henke", name="Si")
platinum = xrt_materials.Material("Pt", rho=21.45, table="Henke", name="Pt")

base_grating_kwargs = dict(
    period_lpermm=400,
    width_to_period_ratio=0.67,
    substrate_material=silicon,
    layer_material=platinum,
    layer_thickness_nm=28.77,
    left_wall_angle_deg=15.0,
    right_wall_angle_deg=15.0,
    x_resolution_nm=0.5,
    z_resolution_nm=0.1,
)

energy_ev = 1000.0
diffraction_order = 1
cff = 2.25
depths_nm = np.arange(10.0, 31.0, 1.0)

parser = argparse.ArgumentParser(description="Run batch simulation with user cases")
parser.add_argument(
    "--quick",
    action="store_true",
    help="Run with lower Fourier order for quick testing",
)
output_dir = Path(__file__).resolve().parent / "results"

args = parser.parse_args()
output_dir.mkdir(parents=True, exist_ok=True)

quick_mode = args.quick
checkpoint_dir = output_dir / "checkpoints_depth_sweep"

grazing_angle_deg = float(
    rp.monochromator_grazing_angles_deg(
        [energy_ev],
        period_lpermm=base_grating_kwargs["period_lpermm"],
        diffraction_order=diffraction_order,
        cff=cff,
    )[0]
)

if quick_mode:
    print("Quick mode: full depth range, lower Fourier order")
else:
    print("Full mode: full depth range, production Fourier order")

user_cases = []
for depth_nm in depths_nm:
    grating = rp.LaminarGrating(
        depth_nm=float(depth_nm),
        **base_grating_kwargs,
    )
    user_cases.append(
        {
            "case_id": f"user-laminar-depth-{int(depth_nm):03d}",
            "label": f"Laminar grating at depth {depth_nm:.1f} nm",
            "grating": grating,
            "energy_ev": energy_ev,
            "grazing_angle_deg": grazing_angle_deg,
            "diffraction_order": diffraction_order,
            "depth_nm": float(depth_nm),
        }
    )

default_fourier = 5 if quick_mode else 25

runner = rp.BatchSimulationRunner(
    default_diffraction_order=diffraction_order,
    default_fourier_orders=default_fourier,
    show_progress=True,
    live_plot=False,
    on_error="continue",
    checkpoint_dir=checkpoint_dir,
    checkpoint_interval=1,
    resume=False,
)

results = list(runner.run_cases(user_cases))

csv_path = output_dir / "batch_user_cases_all_orders.csv"
rp.write_all_orders_csv(results, csv_path)

orders_plot_path = output_dir / "batch_user_cases_orders_1_3_vs_depth.png"
depth_values = np.asarray([float(result.case_data["depth_nm"]) for result in results], dtype=float)
figure, axis = plt.subplots(figsize=(10, 6))
markers = ["o", "s", "^"]
for index, order in enumerate([1, 2, 3]):
    order_efficiency = np.asarray(
        [
            rp.efficiency_for_order(
                result.orders,
                result.efficiency_all,
                diffraction_order=order,
            )
            for result in results
        ],
        dtype=float,
    )
    axis.plot(
        depth_values,
        order_efficiency,
        f"{markers[index]}-",
        linewidth=1.0,
        markersize=3.0,
        label=f"Order {order}",
    )
axis.set_xlabel("Grating Depth (nm)")
axis.set_ylabel("Diffraction Efficiency")
axis.set_title("Batch User Cases: Orders 1-3 Efficiency vs Depth at 1000 eV")
axis.grid(True, alpha=0.3)
axis.legend(loc="best")
figure.tight_layout()
figure.savefig(orders_plot_path, dpi=150, bbox_inches="tight")
plt.close(figure)

profile_grating = rp.LaminarGrating(depth_nm=depths_nm[0], **base_grating_kwargs)
profile_path = output_dir / "batch_user_cases_profile.png"
profile_grating.plot_profile(profile_path)

print(f"Results saved to: {csv_path}")
print(f"Orders 1-3 plot saved to: {orders_plot_path}")
print(f"Profile plot saved to: {profile_path}")
print(f"Energy: {energy_ev:.1f} eV")
print(f"Monochromator grazing angle (cff={cff}): {grazing_angle_deg:.6f} deg")
print(f"Checkpoint directory: {checkpoint_dir}")
