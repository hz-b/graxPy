import numpy as np
import pandas as pd

from pathlib import Path
import grax as rp

rp.setup_logging(level='INFO', run_id='blazed_sweep')


example_root = Path(__file__).resolve().parent
repo_root = Path(__file__).resolve().parents[2]
optical_constants_dir = example_root / "optical_constants"
results_dir = Path(__file__).resolve().parent / "results"
results_dir.mkdir(parents=True, exist_ok=True)
csv_path = results_dir / "blazed_comparison_monochromator_orders_1_3.csv"
plot_path = results_dir / "blazed_comparison_monochromator_orders_1_3.png"
profile_plot_path = results_dir / "blazed_comparison_profile.png"

# Import Optical Constants
silicon = pd.read_csv(
    optical_constants_dir / "OC_Si_SSTR.dat",
    # skiprows=1,
    sep=r"\s*,\s*|\s+",
    engine="python",
)
silicon.attrs["name"] = "Si"
gold = pd.read_csv(
    optical_constants_dir / "OC_Au_SSTR.dat",
    # skiprows=1,
    sep=r"\s*,\s*|\s+",
    engine="python",
)
gold.attrs["name"] = "Au"

# Define Grating

period_lpermm = 600
blaze_angle_deg = 0.729
anti_blaze_angle_deg = 5.597


grating = rp.BlazedGrating(
    period_lpermm=600,
    blaze_angle_deg=0.729,
    anti_blaze_angle_deg=5.597,
    substrate_material=silicon,
    layer_material=gold,
    layer_thickness_nm=30.0,
    top_cap_material=None,
    top_cap_thickness_nm=0.0,
    z_resolution_nm=0.1,
    x_resolution_nm=0.1,
)

# Energy range
energies_ev = np.arange(50.0, 2000.0, 10.0)

# Use monochromator_cases helper
cases = rp.monochromator_cases(
    grating=grating,
    energies_ev=energies_ev,
    period_lpermm=period_lpermm,
    diffraction_order=1,
    cff=2.25,
)

runner = rp.BatchSimulationRunner(
    default_diffraction_order=1,
    default_fourier_orders=20,
    show_progress=True,
    live_plot=True,
    live_plot_x_key="energy_ev",
    live_plot_order_count=3,
    checkpoint_dir=results_dir / "checkpoints",
    resume=False,  # Set to False to force restart from beginning
)

grating.plot_profile(profile_plot_path)
batch_result = list(runner.run_cases(
    cases,
    metadata={
        "grating_type": "BlazedGrating",
        "period_lpermm": period_lpermm,
        "blaze_angle_deg": blaze_angle_deg,
        "anti_blaze_angle_deg": anti_blaze_angle_deg,
        "substrate_material": "Si",
        "layer_material": "Au",
        "layer_thickness_nm": 30.0,
        "fourier_orders": 20,
        "solver_backend": "s_matrix",
        "description": "Blazed grating monochromator sweep",
    }
))
rp.write_all_orders_csv(batch_result, csv_path)
rp.plot_order_subset(
    batch_result,
    plot_path,
    diffraction_orders=[1, 2, 3],
    title="Blazed Grating Monochromator Sweep (600 l/mm, BA=0.729 deg): Orders 1-3",
)

print(f"Computed {sum(case.status == 'ok' for case in batch_result)} monochromator points.")
print(f"Profile plot saved to: {profile_plot_path}")
print(f"Monochromator sweep CSV saved to: {csv_path}")
print(f"Monochromator orders plot saved to: {plot_path}")
