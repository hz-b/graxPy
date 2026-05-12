"""Generate a custom sinusoidal grating profile plot."""

from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import grax as rp
from grax.gratings import BaseGrating
from xrt.backends.raycing import materials as xrt_materials


class SinusoidalGrating(BaseGrating):
    """Sinusoidal custom grating profile."""

    def __init__(self, *, depth_nm: float = 20.0, samples_per_period: int = 1024, **kwargs):
        super().__init__(**kwargs)
        self.depth_nm = float(depth_nm)
        self.samples_per_period = int(samples_per_period)

    def profile_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Return one sinusoidal period."""

        x_positions = np.linspace(0.0, self.period_nm, self.samples_per_period + 1, dtype=float)
        heights = 0.5 * self.depth_nm * (
            1.0 + np.sin(2.0 * np.pi * x_positions / self.period_nm)
        )
        return x_positions, heights

    def profile_depth_nm(self) -> float:
        """Return sinusoidal peak-to-valley depth."""

        return self.depth_nm


silicon = xrt_materials.Material("Si", rho=2.329, table="Henke", name="Si")
gold = xrt_materials.Material("Au", rho=19.3, table="Henke", name="Au")

grating = SinusoidalGrating(
    period_lpermm=600,
    depth_nm=30.0,
    samples_per_period=1024,
    substrate_material=silicon,
    layer_material=gold,
    layer_thickness_nm=30.0,
    top_cap_material=None,
    top_cap_thickness_nm=0.0,
    x_resolution_nm=1.0,
    z_resolution_nm=0.1,
)

parser = argparse.ArgumentParser(description="Generate custom sinusoidal grating profile plot")
parser.add_argument(
    "--output-dir",
    type=Path,
    default=None,
    help="Override output directory (default: examples/grating/results/)",
)
args = parser.parse_args()

if args.output_dir:
    output_dir = Path(args.output_dir)
else:
    output_dir = Path(__file__).resolve().parent / "results"

output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "sinusoidal_custom_profile.png"

grating.plot_profile(output_path)
print(f"Saved: {output_path}")
