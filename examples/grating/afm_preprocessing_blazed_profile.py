"""Build a blazed profile grating from a sample AFM line-scan data file."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import grax

results_dir = Path(__file__).resolve().parent / "results" / "afm_preprocessing_blazed"
results_dir.mkdir(parents=True, exist_ok=True)

period_lpermm = 600
period_nm = 1e6 / period_lpermm
afm_file = Path(__file__).resolve().parent / "data" / "afm_profile_example_blazed.txt"
afm_data = np.loadtxt(afm_file)

afm = grax.AFMPreprocessing(
    afm_data,
    units="m",
    results_folder=results_dir,
    show_plots=False,
)
afm.normalize_scan(reverse=True, zero_baseline=True)
afm.find_troughs(period_nm=period_nm, min_separation_fraction=0.4, profile_type="blazed")
afm.extract_period(average=True)
afm.apply_periodicity_ramp()
afm.rescale_period(period_nm=period_nm)

grating = grax.AFMGrating.from_preprocessing(
    afm,
    # period_lpermm is inferred from the processed AFM profile span if omitted.
    # period_lpermm=period_lpermm,
    substrate_material="Si",
    layer_material="Au",
    layer_thickness_nm=30.0,
    x_resolution_nm=2.0,
    z_resolution_nm=1.0,
)

profile_path = results_dir / "afm_preprocessing_blazed_profile.png"
grating.plot_profile(profile_path)

print(f"Profile depth: {grating.profile_depth_nm():.3f} nm")
print(f"Saved profile plot: {profile_path}")
