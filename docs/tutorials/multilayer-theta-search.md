# Multilayer theta search

Use {func}`grax.run_multilayer_theta_search_sweep` when each energy point
needs its own grazing-angle optimization before the final RCWA solve.

The sweep internally performs, for each energy:

1. a rough scan around an estimated center,
2. a fine scan around the rough maximum,
3. a final solve at the selected grazing angle.

```python
from pathlib import Path
import logging

import numpy as np
import pandas as pd

import grax

example_root = Path("examples/simulation/multilayer_theta_search")
output_dir = example_root / "results"
optical_constants_dir = example_root / "optical_constants"
output_dir.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)

energies_ev = np.linspace(1200.0, 2400.0, 7, dtype=float)

silicon = pd.read_csv(optical_constants_dir / "OC_Si_SSTR.dat", sep=r"\s*,\s*|\s+", engine="python")
silicon.attrs["name"] = "Si"
chromium = pd.read_csv(optical_constants_dir / "OC_Cr_SSTR.dat", sep=r"\s*,\s*|\s+", engine="python")
chromium.attrs["name"] = "Cr"
carbon = pd.read_csv(optical_constants_dir / "OC_C_SSTR.dat", sep=r"\s*,\s*|\s+", engine="python")
carbon.attrs["name"] = "C"

multilayer_stack = grax.MultilayerStack(
    substrate_material=silicon,
    material_a=chromium,
    material_b=carbon,
    d_period_nm=4.8,
    gamma=0.4,
    n_bilayers=60,
    top_material=carbon,
)

grating = grax.BlazedGrating(
    period_lpermm=2400,
    blaze_angle_deg=1.37,
    anti_blaze_angle_deg=3.25,
    coating_stack=multilayer_stack,
    x_resolution_nm=1.0,
    z_resolution_nm=1.0,
)

sweep = grax.run_multilayer_theta_search_sweep(
    grating=grating,
    energies_ev=energies_ev,
    output_dir=output_dir,
    diffraction_order=2,
    multilayer_bragg_order=1,
    rough_scan_half_width_deg=5,
    rough_scan_points=31,
    rough_fourier_orders=2,
    rough_x_resolution_nm=1.0,
    rough_z_resolution_nm=1.0,
    fine_scan_half_width_deg=2,
    fine_scan_points=41,
    fine_fourier_orders=3,
    fine_x_resolution_nm=0.5,
    fine_z_resolution_nm=0.5,
    final_fourier_orders=15,
    final_x_resolution_nm=0.05,
    final_z_resolution_nm=0.05,
    precise_peak_selection_mode="voigt",
    retry_on_selected_efficiency_zero=True,
    retry_selected_efficiency_threshold=0.001,
    max_zero_efficiency_retries=3,
    max_workers="auto",
    show_progress=True,
    live_plot=True,
    on_error="fail_fast",
    checkpoint_dir=output_dir / "checkpoints",
    resume=True,
    theta_tracking_mode="auto",
    max_tracking_energy_step_ev=None,
    backend="numba",
)

logger.info("Sweep completed successfully.")
print(f"Summary saved to: {sweep.summary_csv_path}")
print(f"All-order CSV saved to: {sweep.all_orders_csv_path}")
print(f"Final efficiency plot saved to: {sweep.energy_efficiency_plot_path}")
print(f"Workflow plot saved to: {sweep.workflow_plot_path}")
print(f"Total accumulated scan time: {sweep.total_elapsed_seconds:.1f} s")
if sweep.profile_plot_path is not None:
    print(f"Profile plot saved to: {sweep.profile_plot_path}")
if sweep.stack_plot_path is not None:
    print(f"Stack schematic saved to: {sweep.stack_plot_path}")
print(f"Per-energy theta scans saved to: {sweep.theta_scan_directory}")
logger.info("Outputs written under: %s", output_dir)
```

```{image} images/simulation/multilayer_theta_search_workflow.png
:alt: Multilayer theta-search workflow showing final efficiency and first-energy scan diagnostics
:align: center
:width: 90%
```

See `examples/simulation/multilayer_theta_search/multilayer_theta_search.py`
for the full runnable script.
