# AFM Preprocessing to ProfileGrating

This tutorial shows the AFM preprocessing pipeline with sample AFM line scans
and builds `AFMGrating` objects that can be simulated with the normal `grax`
workflow. The repository now ships two checked-in examples:

- a blazed AFM workflow
- a laminar AFM workflow

Both follow the same preprocessing pipeline, but they use different trough
detection modes and save their diagnostics into separate results folders.

## Pipeline overview

### Blazed example

```python
import numpy as np
import grax

period_lpermm = 600
period_nm = 1e6 / period_lpermm
afm_file = "examples/grating/data/afm_profile_example_blazed.txt"
afm_data = np.loadtxt(afm_file)

afm = grax.AFMPreprocessing(
    afm_data,
    units="m",
    results_folder="examples/grating/results/afm_preprocessing_blazed",
    show_plots=False,
)
afm.normalize_scan(reverse=True, zero_baseline=True)
afm.find_troughs(
    period_nm=period_nm,
    min_separation_fraction=0.4,
    profile_type="blazed",
)
afm.extract_period(average=True)
afm.apply_periodicity_ramp()
afm.rescale_period(period_nm=period_nm)

grating = grax.AFMGrating.from_preprocessing(
    afm,
    # period_lpermm can be inferred from the processed profile span
    # period_lpermm=period_lpermm,
    substrate_material="Si",
    layer_material="Au",
    layer_thickness_nm=30.0,
)
```

### Laminar example

```python
import numpy as np
import grax

period_lpermm = 600
period_nm = 1e6 / period_lpermm
afm_file = "examples/grating/data/afm_profile_example_laminar.txt"
afm_data = np.loadtxt(afm_file)

afm = grax.AFMPreprocessing(
    afm_data,
    units="m",
    results_folder="examples/grating/results/afm_preprocessing_laminar",
    show_plots=False,
)
afm.normalize_scan(reverse=True, zero_baseline=True)
afm.find_troughs(
    period_nm=period_nm,
    min_separation_fraction=0.4,
    profile_type="laminar",
)
afm.extract_period(average=True)
afm.apply_periodicity_ramp()
afm.rescale_period(period_nm=period_nm)

grating = grax.AFMGrating.from_preprocessing(
    afm,
    substrate_material="Si",
    layer_material="Au",
    layer_thickness_nm=30.0,
)
```

## Step-by-step preprocessing

The preprocessing pipeline saves four diagnostic figures by default in
`results/afm_preprocessing`. The checked-in examples override that path so the
blazed and laminar runs do not overwrite each other:
`examples/grating/results/afm_preprocessing_blazed` and
`examples/grating/results/afm_preprocessing_laminar`. Each figure corresponds
to one explicit pipeline step.

### Blazed diagnostics

#### Step 1: `normalize_scan(reverse=True, zero_baseline=True)`

This step flips the scan direction (if needed) and shifts the baseline so the
minimum groove height is at `z = 0`. Using a common baseline makes trough
detection and period extraction more robust and easier to interpret.

![Blazed step 1 normalize scan](../images/gratings/afm_preprocessing_blazed/01_normalize_scan.png)

#### Step 2: `find_troughs(period_nm=..., min_separation_fraction=0.4, profile_type="blazed")`

This identifies groove-bottom candidates using a minimum separation based on
the expected period. The blazed workflow uses `profile_type="blazed"` so the
red dashed markers come from the local-minimum detection path.

![Blazed step 2 find troughs](../images/gratings/afm_preprocessing_blazed/02_find_troughs.png)

#### Step 3: `extract_period(average=True)`

This interpolates every detected trough-to-trough segment onto a common
normalized x-grid and averages them. Dashed curves are individual periods
(one color per segment); the solid red curve is the averaged unit-cell profile
used downstream.

![Blazed step 3 averaged period extraction](../images/gratings/afm_preprocessing_blazed/03_extract_period_averaged.png)

#### Step 4: `apply_periodicity_ramp()`

This optional correction enforces periodic endpoint continuity by removing a
linear offset between `h(0)` and `h(T)`. Use it when strict periodic boundary
consistency is needed for RCWA.

![Blazed step 4 periodicity ramp](../images/gratings/afm_preprocessing_blazed/04_periodicity_ramp.png)

### Laminar diagnostics

#### Step 1: `normalize_scan(reverse=True, zero_baseline=True)`

The laminar example uses the same normalization step before switching to the
laminar trough-detection mode.

![Laminar step 1 normalize scan](../images/gratings/afm_preprocessing_laminar/01_normalize_scan.png)

#### Step 2: `find_troughs(period_nm=..., min_separation_fraction=0.4, profile_type="laminar")`

The laminar workflow uses `profile_type="laminar"`. In that mode the red dashed
markers are placed at the midpoint between consecutive vertical walls instead of
following an edge minimum on the flat-bottom valley.

![Laminar step 2 find troughs](../images/gratings/afm_preprocessing_laminar/02_find_troughs.png)

#### Step 3: `extract_period(average=True)`

As in the blazed case, the detected trough-to-trough segments are normalized and
averaged into one reusable unit-cell profile.

![Laminar step 3 averaged period extraction](../images/gratings/afm_preprocessing_laminar/03_extract_period_averaged.png)

#### Step 4: `apply_periodicity_ramp()`

The same periodicity correction can be applied after the laminar average profile
is extracted.

![Laminar step 4 periodicity ramp](../images/gratings/afm_preprocessing_laminar/04_periodicity_ramp.png)

## Notes

- Use `average=True` in `extract_period()` when scan noise creates visible
  period-to-period variation.
- Use `profile_type="laminar"` when the AFM scan represents a laminar grating.
  That mode finds the vertical walls first and places the trough at the middle
  of each wall-bounded valley.
- Increase `min_prominence_fraction` in `find_troughs()` when laminar or noisy
  scans contain shallow secondary minima that should not count as full periods.
- Use `apply_periodicity_ramp()` when the extracted period endpoints differ and
  you need strict periodic continuity for RCWA.
- By default, preprocessing plots are saved to
  `./results/afm_preprocessing`. Set `save_plots=False` to disable file output,
  or pass `results_folder=...` to isolate one workflow's diagnostics.
- `AFMGrating.from_preprocessing(...)` accepts the full preprocessing object so
  the exact validated profile state is used directly. `period_lpermm` may be
  passed explicitly or inferred from the extracted profile span.
- Simulation material inputs must be real optical-constants objects such as xrt
  `Material` instances or DataFrame-like optical-constants tables. Bare strings
  such as `"Si"` and `"Au"` are labels only and are not accepted for
  simulation.
- The `03_extract_period_averaged` plot shows all individual normalized periods
  as dashed colored traces plus the solid averaged period profile.
- The full runnable blazed script lives at
  `examples/grating/afm_preprocessing_blazed_profile.py`.
- The matching laminar smoke-test script lives at
  `examples/grating/afm_preprocessing_laminar_profile.py` and uses
  `examples/grating/data/afm_profile_example_laminar.txt`.
