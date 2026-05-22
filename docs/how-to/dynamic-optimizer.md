# Dynamic Optimizer Specs

Use this workflow when you want to fit a grating that is not hard-coded in
`grax_opt`, or when you want to couple parameters together without changing the
library source.

The dynamic optimizer accepts a plain Python mapping that defines:

- which parameters are free
- the bounds for each parameter
- optional equality ties between parameters
- a custom `build_grating` callable

## Example

```python
from pathlib import Path

from grax import LaminarGrating
from grax_opt import optimize_dynamic
import pandas as pd


optical_constants_dir = Path("examples/optical_constants")
silicon = pd.read_csv(
    optical_constants_dir / "n_Si_cxro.txt",
    skiprows=1,
    sep=r"\s*,\s*|\s+",
    engine="python",
)
silicon.attrs["name"] = "Si"

platinum = pd.read_csv(
    optical_constants_dir / "n_Pt_cxro.txt",
    skiprows=1,
    sep=r"\s*,\s*|\s+",
    engine="python",
)
platinum.attrs["name"] = "Pt"

carbon = pd.read_csv(
    optical_constants_dir / "n_C_cxro.txt",
    skiprows=1,
    sep=r"\s*,\s*|\s+",
    engine="python",
)
carbon.attrs["name"] = "C"


def build_grating(parameters):
    return LaminarGrating(
        period_lpermm=float(parameters["period_lpermm"]),
        width_to_period_ratio=float(parameters["width_to_period_ratio"]),
        depth_nm=float(parameters["depth_nm"]),
        left_wall_angle_deg=float(parameters["left_wall_angle_deg"]),
        right_wall_angle_deg=float(parameters["right_wall_angle_deg"]),
        substrate_material=silicon,
        layer_material=platinum,
        layer_thickness_nm=28.77,
        top_cap_material=carbon,
        top_cap_thickness_nm=float(parameters["top_cap_thickness_nm"]),
    )


spec = {
    "build_grating": build_grating,
    "parameter_bounds": {
        "period_lpermm": (392.0, 408.0),
        "width_to_period_ratio": (0.60, 0.75),
        "depth_nm": (12.0, 18.0),
        "left_wall_angle_deg": (5.0, 25.0),
        "right_wall_angle_deg": (5.0, 25.0),
        "top_cap_thickness_nm": (0.0, 1.0),
    },
    "equality_constraints": {
        "right_wall_angle_deg": "left_wall_angle_deg",
    },
    "measurement_path": Path("measurement.dat"),
    "output_dir": Path("results/dynamic_fit"),
    "evaluation_energies_ev": [100.0, 150.0, 200.0],
    "evaluation_grazing_angles_deg": [4.0],
}

result = optimize_dynamic(spec)
```

In this example, Ax optimizes `left_wall_angle_deg` only once, and the dynamic
resolver copies that value into `right_wall_angle_deg` before the grating is
built.

`build_grating` must provide materials accepted by
`resolve_refractive_index` (for example a pandas DataFrame with optical
constants, or an object exposing `get_refractive_index()`).

If you provide `evaluation_grazing_angles_deg`, the optimizer treats the
evaluation inputs as explicit energy-angle cases. One energy can be paired with
many angles, or one angle can be paired with many energies. Any combination
with more than one energy and more than one angle is rejected.

## Constraint Semantics

Equality ties are one-way:

- `right_wall_angle_deg = left_wall_angle_deg`
- `A = B = C` is represented as two ties, for example `B -> A` and `C -> B`

The optimizer rejects cycles and unknown parameter names early, before Ax runs.

## When to use it

Use the dynamic optimizer when:

- you are experimenting with a new grating class or builder
- you want to re-use the optimizer without editing the source tree
- you need parameter coupling such as symmetric wall angles

For the built-in laminar and blazed workflows, the existing dedicated entry
points still work unchanged.
