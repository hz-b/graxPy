# Piezoelectric X-ray pulse picker

The LNB preset implements the geometry reported in Table 1 of [Vadilonga *et al.*, *Optics Express* **29**, 34962 (2021)](https://doi.org/10.1364/OE.438570): a 4 µm deformation period, 2 µm aluminium electrode pitch, 1 µm finger width, and approximately 100 nm Al thickness. The LNB crystalline kernel uses the LiNbO₃ (104) reflection at 8 keV. The LGS preset uses a 2.8 µm deformation period, 1.4 µm electrode pitch, and 0.7 µm fingers. These dimensions follow the table; some prose in the article reverses the period relationship.

There are two distinct calculations:

- **Rocking curve**: `xrt` supplies the complex dynamical Bragg amplitude of the LNB crystal. The periodic deformation contributes a phase `exp(i Q_B u_z(x))`. An RCWA calculation of Al-only transmission supplies a complex electrode screen; the screen is applied to the inward and outward passage. Fourier coefficients of the resulting field scale each Bragg satellite. Energy bandwidth, angular divergence, and detector aperture broaden the computed intensity.
- **RCWA geometry**: the ordinary `grax` RCWA solver sees the continuous optical refractive indices of the deformed substrate and Al. It reports reflected and transmitted orders for deformation-only, Al-only, and combined geometries. It does **not** resolve atomic crystal planes and therefore its reflected efficiency is not the crystalline Bragg efficiency.

The default `displacement_per_volt_nm=0.0104` and effective 8 keV bandwidth of 0.85 eV are fits of this reduced model to raster-digitized Fig. 4, **not** measured material or instrument parameters. The inferred 9 V peak-to-valley deformation is about 0.094 nm. The OFF curve has normalized RMS residual about 0.017; the ON curve remains at about 0.111, with visibly underestimated satellites. The pure sinusoidal thin-screen model is thus a useful diagnostic, **not a quantitative reproduction of every Fig. 4 intensity**. `displacement_harmonics` allows additional cosine harmonics, while the pure sinusoid remains the default.

The LGS preset has zero voltage-to-displacement response until a separate calibration is supplied; the LNB Fig. 4 fit must not be transferred to LGS.

Run the complete maintained example from the repository root:

```bash
python examples/pulse_picker/paper_lnb.py
```

It saves `geometry.png`, `fig4_comparison.png` (with residuals), `rcwa_diagnostic.png`, CSV tables, and `summary.json` under `examples/pulse_picker/results/`. The source image, pixel calibration, uncertainty estimate, and regeneration commands are in `data/pulse_picker/README.md`. Fig. 4 CSV points are raster-derived estimates rather than raw experimental measurements.

To test whether a non-sinusoidal deformation improves the residual while keeping the pure-sinusoid run intact, use a separate output directory, for example `python examples/pulse_picker/paper_lnb.py --harmonics 0.2,-0.1 --output /tmp/pulse-picker-harmonics`. The coefficients are dimensionless weights of the second and third cosine harmonics; the code normalizes the combined profile to the fitted peak-to-valley displacement.

For an individual simulation:

```python
import numpy as np
from grax import (
    PulsePickerGrating,
    pulse_picker_rcwa_amplitudes,
    simulate_pulse_picker_rocking_curve,
)

picker = PulsePickerGrating.paper_preset("LNB")
off = simulate_pulse_picker_rocking_curve(picker.at_voltage(0))
on = simulate_pulse_picker_rocking_curve(picker.at_voltage(9))
orders, reflected_amplitude, transmitted_amplitude, R, T = (
    pulse_picker_rcwa_amplitudes(picker.at_voltage(9), fourier_orders=8)
)
assert np.all(R >= 0) and np.all(T >= 0)
```

`picker.with_component("deformation")` and `picker.with_component("aluminium")` provide the diagnostic single-mechanism geometries. The `PulsePickerGrating` OFF state is `picker.at_voltage(0)`. The LGS rocking model requires `bragg_reference=(angles_deg, complex_amplitudes)` from an independently calibrated crystalline reference; LGS RCWA needs no such reference.

`simulate_pulse_picker_timing` interpolates an efficiency–voltage curve over a configurable effective electrical gate. Its default 1 ns Gaussian FWHM and 2 ns bunch spacing are simulation inputs. The calculated neighbouring-bunch leakage is conditional on that waveform and should **not** be described as a measured switching or extinction performance.

In the local web UI, create a grating of type **Piezoelectric pulse picker**, save it, then choose **Rocking curve** or **RCWA** on its detail page. Both workflows save a PNG plot, CSV data, and a JSON summary for download. The rocking plot overlays Fig. 4's digitalized points at the LNB 8 keV default. LGS can currently use the web RCWA workflow only.
