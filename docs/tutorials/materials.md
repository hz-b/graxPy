# Materials

`grax` accepts material inputs anywhere a grating or multilayer asks for a
material. The material resolver supports two concrete patterns:

- A pandas DataFrame-like object with columns `Energy(eV)`, `Delta`, and `Beta`.
- An xrt Material object.

Bare strings such as `"Si"` or `"Au"` are not accepted as simulation material
inputs. They are treated only as labels in plots or metadata. For simulations,
pass an xrt `Material` object or a DataFrame-like optical-constants table.

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

```python
from xrt.backends.raycing import materials as xrt_materials

platinum = xrt_materials.Material("Pt",
                                rho=20.13,
                                table="Henke", 
                                name="Pt-xrt")
```



At solve time, grax internally interpolates `Delta` and `Beta` at the
requested photon energy and uses `n = 1 - delta + i beta`.
