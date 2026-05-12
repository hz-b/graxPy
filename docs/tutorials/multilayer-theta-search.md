# Multilayer theta search

Use {func}`grax.run_multilayer_theta_search` when each energy point
needs its own grazing-angle optimization before the final RCWA solve.

The workflow for each energy is:

1. estimate the multilayer Bragg angle,
2. run a coarse rough scan around that estimate,
3. run a precise scan around the rough maximum,
4. rerun the final simulation at the selected theta.

The final theta selection from the precise scan is controlled by
`precise_peak_selection_mode`:

- `max`: use the sampled discrete maximum directly
- `gauss`: fit a Gaussian to a local neighborhood around the sampled maximum
- `voigt`: fit a Voigt profile to the same kind of local neighborhood

Only the **precise** scan uses this fitting logic. The rough scan still uses the
sampled maximum to locate the neighborhood for the final refinement.

The fit window is always local, not the whole precise scan:

- the code first looks for half-maximum crossings on both sides of the sampled peak
- when both crossings exist, that peak span is expanded by a small fixed margin
- if the half-maximum crossings are not both available, the code falls back to a
  symmetric odd-sized window centered on the sampled maximum

If the requested fit is not usable, the fallback chain is:

1. requested fit model
2. alternate fit model
3. sampled discrete maximum

For example, `voigt` falls back to `gauss`, and only then to `max`. The final
reported efficiency is still obtained from a fresh RCWA solve at the selected
theta. The fit amplitude itself is not used as the physical efficiency.

This tutorial example reuses the same blazed-multilayer geometry as the
comparison workflow in `comparison_to_other_codes/blazed_multilayer`, but keeps
the numerics intentionally cheap:

- `rough_fourier_orders = 3`
- `fine_fourier_orders = 5`
- `final_fourier_orders = 25`
- `rough_x_resolution_nm = 1.0`
- `rough_z_resolution_nm = 1.0`
- `fine_x_resolution_nm = 0.5`
- `fine_z_resolution_nm = 0.5`
- `final_x_resolution_nm = 0.3`
- `final_z_resolution_nm = 0.3`

```python
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import grax as rp

example_root = Path("examples/simulation/multilayer_theta_search")
output_dir = example_root / "results"
optical_constants_dir = example_root / "optical_constants"
theta_scan_dir = output_dir / "theta_scans"
output_dir.mkdir(parents=True, exist_ok=True)
theta_scan_dir.mkdir(parents=True, exist_ok=True)

energies_ev = np.linspace(1200.0, 2400.0, 7, dtype=float)

silicon = pd.read_csv(optical_constants_dir / "OC_Si_SSTR.dat", sep=r"\s*,\s*|\s+", engine="python")
silicon.attrs["name"] = "Si"
chromium = pd.read_csv(optical_constants_dir / "OC_Cr_SSTR.dat", sep=r"\s*,\s*|\s+", engine="python")
chromium.attrs["name"] = "Cr"
carbon = pd.read_csv(optical_constants_dir / "OC_C_SSTR.dat", sep=r"\s*,\s*|\s+", engine="python")
carbon.attrs["name"] = "C"

multilayer_stack = rp.MultilayerStack(
    substrate_material=silicon,
    material_a=chromium,
    material_b=carbon,
    d_period_nm=4.8,
    gamma=0.4,
    n_bilayers=60,
    top_material=carbon,
)

grating = rp.BlazedGrating(
    period_lpermm=2400,
    blaze_angle_deg=1.37,
    anti_blaze_angle_deg=3.25,
    coating_stack=multilayer_stack,
    x_resolution_nm=0.5,
    z_resolution_nm=0.5,
)

results = []
for energy_ev in energies_ev:
    result = rp.run_multilayer_theta_search(
        grating=grating,
        energy_ev=float(energy_ev),
        diffraction_order=2,
        precise_peak_selection_mode="voigt",
        rough_fourier_orders=3,
        fine_fourier_orders=5,
        final_fourier_orders=25,
        rough_x_resolution_nm=1.0,
        rough_z_resolution_nm=1.0,
        fine_x_resolution_nm=0.5,
        fine_z_resolution_nm=0.5,
        final_x_resolution_nm=0.3,
        final_z_resolution_nm=0.3,
    )
    results.append(result)
    diagnostics = result.theta_search_diagnostics
    energy_tag = f"{int(round(result.energy_ev))}eV"
    with (theta_scan_dir / f"theta_scan_{energy_tag}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["grazing_angle_deg", "selected_efficiency", "is_selected_peak"])
        for grazing_angle_deg, efficiency in zip(
            diagnostics.precise_grazing_angles_deg,
            diagnostics.precise_efficiencies,
        ):
            writer.writerow(
                [
                    grazing_angle_deg,
                    efficiency,
                    int(np.isclose(grazing_angle_deg, diagnostics.selected_grazing_angle_deg)),
                ]
            )

diagnostics = results[0].theta_search_diagnostics
figure, axes = plt.subplots(1, 2, figsize=(12, 5.5))
axes[0].plot(
    [result.energy_ev for result in results],
    [result.selected_efficiency for result in results],
    "o-",
)
axes[0].set_xlabel("Energy (eV)")
axes[0].set_ylabel("Efficiency (2nd order)")

axes[1].plot(diagnostics.rough_grazing_angles_deg, diagnostics.rough_efficiencies, "o-", label="Rough")
axes[1].plot(diagnostics.precise_grazing_angles_deg, diagnostics.precise_efficiencies, "s-", label="Precise")
axes[1].plot(diagnostics.selected_grazing_angle_deg, diagnostics.selected_efficiency, "r*", label="Peak")
axes[1].set_xlabel("Grazing Angle (deg)")
axes[1].set_ylabel("Efficiency (2nd order)")
axes[1].legend(loc="best")

figure.tight_layout()
figure.savefig(output_dir / "multilayer_theta_search_workflow.png", dpi=150, bbox_inches="tight")
plt.close(figure)
```

Each energy also saves its final precise theta scan to `results/theta_scans/`
as both CSV and PNG so you can inspect the selected peak afterwards.

```{image} images/simulation/multilayer_theta_search_workflow.png
:alt: Multilayer theta-search workflow showing final efficiency and first-energy scan diagnostics
:align: center
:width: 90%
```

See `examples/simulation/multilayer_theta_search/multilayer_theta_search.py`
for the full runnable script.
