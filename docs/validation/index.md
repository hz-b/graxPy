# Validation

This section compares `grax` with three external references:

- [RETICOLO v9](https://zenodo.org/records/4419063) (RCWA solver)
- [DiffractMod](https://www.synopsys.com/blogs/chip-design/rsoft-device-tools.html) (RCWA solver)
- [REFLEC](https://www.helmholtz-berlin.de/forschung/oe/wi/optik-strahlrohre/arbeitsgebiete/ray_en.html) (Nevier)

All simulation curves use the same optical constants, and measured data is included for comparison.

`grax` also ships two selectable solver paths of its own, and
[RCWA vs Nevière differential method](solver-comparison.md) runs the same
validation cases through both. Note that REFLEC, one of the external references
above, is itself a differential-method code.

```{toctree}
:maxdepth: 1

blazed
blazed-multilayer
laminar
laminar-150lmm
solver-comparison
```
