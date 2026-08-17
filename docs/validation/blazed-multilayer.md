# Blazed Multilayer Grating

This comparison uses the blazed multilayer workflow from `validation/blazed_multilayer`.

All simulation curves use the same optical constants. The comparison plot includes
the external DiffraMod reference together with two grax workflows applied to
the same second diffraction order.

## Grating design parameters

- Grating type: `BlazedGrating`
- Period: `2400 l/mm`
- Blaze angle: `1.37 deg`
- Anti-blaze angle: `3.25 deg`
- Substrate material: `Si`
- Multilayer: `Cr/C`
- Bilayer period: `4.8 nm`
- Bilayer count: `60`
- Top material: `C`

## Multilayer stack

```{image} images/multilayer_stack_schematic.png
:alt: Blazed multilayer stack schematic
:align: center
:width: 65%
```

## Grating profile

```{image} images/blazed_multilayer_profile.png
:alt: Blazed multilayer grating profile
:align: center
:width: 85%
```

## Results

The three curves in the comparison figure are obtained as follows:

- `DiffraMod`: reference efficiencies from
  `validation/blazed_multilayer/simulation/DiffractMod_CrC_d4.8_N60.dat`
- `grax energy-angle`: the grax sweep in
  `validation/blazed_multilayer/run_rcwa.py`, where
  each grax simulation uses the energy and grazing angle pair taken directly
  from the DiffraMod table
- `grax theta-search`: the adaptive multilayer theta-search workflow, where
  grax determines the grazing angle internally at each energy by running a
  rough scan, a precise scan, and a final solve at the selected peak

This means the simulations in the `simulation/` folder that follow the DiffraMod
angles are not performing an internal angle search; they reuse the grazing angles
already present in the DiffraMod dataset. The theta-search curve is the one that
solves for the angle inside grax itself.

```{image} images/comparison_blazed_multilayer_sweep.png
:alt: Blazed multilayer comparison
:align: center
:width: 85%
```

For the low-energy region, the same three curves are also shown in a zoomed view
between `550 eV` and `600 eV`, with the vertical axis limited to `0.4`.

```{image} images/comparison_blazed_multilayer_sweep_550_600eV.png
:alt: Blazed multilayer comparison zoomed to 550-600 eV
:align: center
:width: 85%
```

To make the angle-selection differences explicit, the figure below compares the
grazing angle used by `DiffractMod` (`alpha` in the source table) with the
angle selected by `grax` theta-search at each energy.

```{image} images/comparison_blazed_multilayer_grating_angle.png
:alt: Blazed multilayer comparison of grazing angle versus energy
:align: center
:width: 85%
```

The lower second-order intensity seen in the `grax` theta-search curve is due
to numerical resolution settings (Fourier-order truncation and `x`/`z`
discretization), not to a different grazing-angle choice. The compared runs use
the same grazing angles. When the DiffraMod angles are reused in `grax` and the
simulation is run with higher fine/final Fourier order and finer fine/final
`x`/`z` resolution,
`grax` reproduces the same results.
