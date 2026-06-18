# Materials

`grax` accepts material inputs anywhere a grating or multilayer asks for a material. The resolver supports four
patterns, in this order:

- A string elemental material name such as `"Si"`, `"Pt"`, or `"Au"` when a packaged Henke table exists for that
  symbol.
- A `grax.MaterialSpec` with `name` and optional `density_g_cm3` when you want to override the default density.
- A pandas DataFrame-like object with columns `Energy(eV)`, `Delta`, and `Beta`.
- An xrt-compatible object with `get_refractive_index()`.

String material names are the recommended default for elemental Henke lookups. DataFrame-like optical-constants tables
remain supported for custom workflows, and xrt material objects still work for backward compatibility during the
transition. When xrt objects are used, `grax` emits a `FutureWarning` so the deprecation is visible at runtime.

## String material names

For common elemental materials, you can pass the material symbol directly:

```python
import grax

grating = grax.LaminarGrating(
    substrate_material="Si",
    layer_material="Pt",
    top_cap_material="C",
)
```

At solve time, `grax` loads the packaged Henke table for the symbol, converts it to `Delta` and `Beta` using the
built-in density metadata for that element, and interpolates `n = 1 - delta + i beta` at the requested photon energy.

If a material name is not available, the error includes the full supported-material list.

## Density overrides

When you know the actual sample density, use `grax.MaterialSpec` so the override travels with the material input:

```python
import grax

gold_film = grax.MaterialSpec("Au", density_g_cm3=19.0)

grating = grax.BlazedGrating(
    substrate_material="Si",
    layer_material=gold_film,
)
```

This is useful for thin films, porous coatings, and other samples that do not behave like bulk material. The Web UI now
prefills the density fields with the built-in default values; edit them only when your sample density differs from the
tabulated bulk value.

## Built-in Henke density table

The following table is generated from the same runtime density registry that the string-material resolver and the Web
UI use.

```{include} generated/material_density_table.md
```

## Optical constants from files

Most examples load CXRO-style optical constants into pandas DataFrames:

```python
import pandas as pd

platinum = pd.read_csv(
    optical_constants_dir / "Pt_cxro.txt",
    skiprows=1,
    sep=r"\s*,\s*|\s+",
    engine="python",
)
platinum.attrs["name"] = "Pt-cxro"
```

The name stored in `attrs["name"]` is used for plot labels and debug output.

## xrt compatibility

```python
from xrt.backends.raycing import materials as xrt_materials

platinum = xrt_materials.Material(
    "Pt",
    rho=21.45,
    table="Henke",
    name="Pt-xrt",
)
```

This path still works for compatibility, but `grax` warns that xrt material support is deprecated and will be removed
in a future version. For new code, prefer string material names or `grax.MaterialSpec`.

## Shared material list

The Web UI and the Python API use the same packaged Henke material list and density registry. If you need the
available elemental symbols programmatically, use `grax.available_material_symbols()`. If you need the built-in
density table, use `grax.material_density_catalog()`.
