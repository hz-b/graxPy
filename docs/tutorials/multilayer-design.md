# Multilayer grating design workflow

Use {class}`grax.MultilayerGratingDesigner` to size a periodic multilayer coating
for a blazed grating working in a chosen diffraction order. The workflow has two
steps that share one {class}`grax.MultilayerDesignConfig`: a **survey** across
bilayer d-spacing and blaze angle, then an **energy scan** of the designs you
pick off it.

## Quick start: the Ru/B4C example

`examples/simulation/multilayer_grating_design/` is a runnable project that
sizes a Ru/B4C coating on silicon for a 2400 l/mm blazed grating working in
inside order 2 at 9 keV. The images below are its actual output.

### Step 1 -- survey a d-spacing / blaze-angle grid

```console
$ cd examples/simulation/multilayer_grating_design
$ python 0_run_survey.py
```

```{figure} images/simulation/multilayer_design_efficiency_heatmap_d_vs_blaze.png
:alt: Peak selected-order efficiency over the bilayer d-spacing and blaze-angle grid, with the optimal-blaze ridge overlaid.
:align: center
:width: 85%

Peak efficiency over the full `(d, blaze)` grid, with the optimal blaze angle
per d-spacing overlaid as a ridge. One glance shows both where the resonance
sits and how sharp it is -- here a narrow band around d = 3 nm, order-2 clear.
```

### Step 2 -- scan the best design over energy

```console
$ python 1_run_energy_scan.py --best
```

```{figure} images/simulation/multilayer_design_efficiency_vs_energy_best_design.png
:alt: Selected-order efficiency versus photon energy for the survey's best design, d = 3.102 nm, blaze = 0.859 deg.
:align: center
:width: 75%

The survey's best cell, d = 3.102 nm / blaze = 0.859 deg, swept from 3 keV to
12 keV. `theta*` is re-solved at every energy, so this plot is not an
extrapolation from the single survey point -- it is 1000 independent theta
searches.
```

Read on for what each step is doing, how to read the rest of the output, and
how to run this example (or your own) sensibly.

## Step 1: the d-spacing / blaze-angle survey

{meth}`grax.MultilayerGratingDesigner.run_survey` scans a 2-D grid of bilayer
d-spacing (`d_min_nm`..`d_max_nm`, `d_points` values) against blaze angle
(`blaze_min_deg`..`blaze_max_deg`, `blaze_points` values). For every `(d, blaze)`
pair it:

1. builds the multilayer-coated blazed grating with the fixed `gamma`,
   `n_bilayers` and `anti_blaze_angle_deg`, and
2. runs {func}`grax.run_multilayer_theta_search_sweep` at the single
   `target_energy_ev`, using `config.survey_scan_settings`, with `output_dir`
   set to that pair's own run folder. The search scans the incident angle
   around the analytical multilayer Bragg estimate and returns the angle
   `theta*` that maximizes the selected-order efficiency, plus the
   precise-scan FWHM.

The incident angle is not a parameter you choose: every pair gets its own theta
search, and `theta*` is whatever angle that search settles on. Step 2 then
tracks how `theta*` evolves with energy for the designs you pick.

`survey_scan_settings` and `energy_scan_settings` are independent
{class}`grax.ThetaSearchScanSettings` instances on `MultilayerDesignConfig` --
tune the survey's (many single-energy searches, one per grid cell) separately
from the energy scan's (fewer designs, but often many energies each).

The survey writes:

- `survey/survey.csv` -- one row per cell: `d_nm`, `blaze_deg`,
  `bragg_estimate_deg`, `incidence_angle_deg`, `peak_efficiency`,
  `precise_fwhm_deg`, `edge_clipped`.
- `plots/optimal_blaze_vs_d_spacing.png` -- **headline curve 1**: for each
  d-spacing, the blaze angle with the highest efficiency
  (`optimal_blaze_deg = argmax_blaze efficiency`), plotted d-spacing (x) against
  optimal blaze angle (y), each point labelled with its peak efficiency.
- `plots/max_efficiency_vs_d_spacing.png` -- **headline curve 2**: the same
  optimum plotted d-spacing (x) against max selected-order efficiency (y), each
  point labelled with the blaze angle that achieved it.
- `plots/efficiency_heatmap_d_vs_blaze.png` -- **headline map**: the full
  `efficiency_map`, not just the per-d optimum -- d-spacing (x), blaze angle
  (y), peak selected-order efficiency as colour (z) -- with the
  `optimal_blaze_deg` ridge overlaid, so you can see the whole resonance
  structure the two curves above are slices of. Shown above, in *Quick start*.
- `survey/runs/` -- one folder per d-spacing (`d<d>nm/`), each holding an
  `overlay.png` (every blaze angle's incidence-angle scan at that d, with the
  chosen blaze drawn bold and a star at each selected peak) and one sub-folder
  per blaze angle (`blaze<b>deg/`). Each blaze folder is the **full output of that
  multilayer theta search**, exactly as {func}`grax.run_multilayer_theta_search_sweep`
  writes it: `multilayer_theta_search_summary.csv` (selected angle, efficiency,
  FWHM), `multilayer_theta_search_all_orders.csv` (every diffraction order),
  `theta_scans/theta_scan_<E>eV.csv` + `.png` (the incidence-angle scan), the
  profile and stack plots, and `checkpoints/`. A `search_parameters.json` is
  added listing everything that was fed into the search so you can review a run
  for correctness and change its parameters.

The two curves the heatmap is sliced from look like this on the real Ru/B4C
survey (a full 50 x 28 grid, d = 2-8 nm):

```{figure} images/simulation/multilayer_design_optimal_blaze_vs_d_spacing.png
:alt: Optimal blaze angle versus d-spacing, points coloured and labelled by peak efficiency.
:align: center
:width: 85%

Optimal blaze angle per d-spacing. The step pattern between d = 4 and 6 nm is
a real feature of this grid, not noise -- see below.
```

```{figure} images/simulation/multilayer_design_max_efficiency_vs_d_spacing.png
:alt: Max selected-order efficiency versus d-spacing, points coloured and labelled by optimal blaze angle.
:align: center
:width: 85%

The same optimum, efficiency instead of angle on the y-axis. The main
resonance around d = 3 nm reaches ~0.69; efficiency collapses to near zero for
d = 4-6 nm before a much weaker secondary rise past it.
```

A cell whose search raises (for example an unphysical sub-grazing incidence
angle) is recorded as a `NaN` row when `on_error="continue"` (the default) and
re-raises when `on_error="fail_fast"`.

{meth}`grax.MultilayerGratingDesigner.evaluate_survey` (the example's
`0_run_survey.py --eval`) runs **no new solves**: it re-reads every per-run
`multilayer_theta_search_summary.csv` already on disk, rebuilds `survey.csv` and
regenerates all three headline plots and every overlay. Use it to iterate on
the analysis after a long `run_survey`.

## Step 2: per-design energy scans

Pick one or more `(d_spacing_nm, blaze_angle_deg)` designs off the survey and
pass them to {meth}`grax.MultilayerGratingDesigner.run_energy_scan`. The example's
`1_run_energy_scan.py` selects them from `survey.csv`: no flag scans the optimal
blaze at every d-spacing, `--best` scans only the single highest-efficiency
`(d, blaze)`, and `--pairs "d,blaze; ..."` scans exactly what you list. Each
design is one fixed grating swept over
`energy_scan_min_ev`..`energy_scan_max_ev` (`energy_scan_points` values) with
{func}`grax.run_multilayer_theta_search_sweep`, giving efficiency versus energy
and `theta*(E)` -- shown for the example's best design in *Quick start* above.
Adding `--eval` ({meth}`grax.MultilayerGratingDesigner.evaluate_energy_scan`)
re-reads existing `energy_scan/` results instead of re-solving.

Besides the sweep's own generic-titled plot (`EnergyScanResult.energy_efficiency_plot_path`,
inside that design's `energy_scan/` folder), every design also gets a plot in
the shared `plot_dir` (`EnergyScanResult.titled_plot_path`, the same folder as
the survey's three headline plots) -- the same curve, titled with the coating
and this design's d-spacing and blaze angle, e.g. *"Ru/B4C multilayer grating
(order 2): d = 3.102 nm, blaze = 0.859 deg"*, and named accordingly:
`efficiency_vs_energy_Ru-B4C_order2_d3.102nm_blaze0.859deg.png`. Because every
design's plot lands in the same folder, the filename itself carries the
coating, order, d-spacing and blaze angle -- there is nothing per-design about
the folder to disambiguate them.

Scanning two or more designs in one call also writes an overlay comparison --
every curve on one axis, legended by d-spacing and blaze angle, as
`plots/efficiency_vs_energy_comparison_<materials>_order<n>.png`.
{meth}`grax.MultilayerGratingDesigner.run_energy_scan` and
{meth}`grax.MultilayerGratingDesigner.evaluate_energy_scan` call
{meth}`grax.MultilayerGratingDesigner.plot_energy_scan_overlay` for you; call it
directly to overlay an arbitrary subset.

`run_energy_scan` also accepts `should_continue`, checked before each design, so
a long scan can be stopped cooperatively; it returns the designs completed so
far. A `stop_event` stops it immediately instead -- see *Progress and abort
hooks* below.

The coating name comes from `MultilayerDesignConfig.coating_label` when set,
otherwise from `"<material_a name>/<material_b name>"` -- set it explicitly
whenever a material is modelled with a stand-in optical-constants table (B4C
modelled with the carbon table, in the Ru/B4C example) so the title and
filename show the real compound.

## Walking through the Ru/B4C example in depth

### The four files

| File | Role |
| --- | --- |
| `rub4c_design_parameters.py` | **The only file you edit.** Builds `CONFIG`; nothing ever rewrites it. |
| `0_run_survey.py` | Step 1. `--eval` re-derives the survey from disk without solving. |
| `1_run_energy_scan.py` | Step 2. `--best`, `--pairs "d,blaze; ..."`, `--eval`. |
| `run_all.sh` | Both steps in order (`PYTHON_BIN` overrides the interpreter). |

The parameters file is split by three `# ===` banners that mirror
`MultilayerDesignConfig`'s own grouping -- `SHARED`, `SURVEY`, `ENERGY_SCAN`.
Check the banner before moving a value: editing the wrong section is the usual
cause of "I changed X and nothing happened". `SURVEY_SCAN.rough_scan_points`
only affects `0_run_survey.py`; `1_run_energy_scan.py` reads `ENERGY_SCAN_SCAN`.

`COATING_LABEL = "Ru/B4C"` is there because B4C is modelled with the carbon
optical-constants table. Without it every plot title and filename would say
"Ru/C", the table's element name rather than the real compound.

### Sizing the grid

The shipped grid is `D_POINTS = 50` x `BLAZE_POINTS = 28` = 1400 theta
searches, which is an overnight run -- that is the grid the images above come
from. For a first pass set both to 2-3 and cut `rough_scan_points`/
`fine_scan_points` to ~15 in **both** scan blocks; the whole thing then
finishes in a couple of minutes and still exercises every artifact path.

### Reading the survey output tree

```text
results/
  survey/survey.csv                     one row per (d, blaze) cell
  survey/runs/d3.102nm/overlay.png      every blaze angle at this d, chosen one bold
  survey/runs/d3.102nm/blaze0.859deg/   the full theta search for one cell
  plots/optimal_blaze_vs_d_spacing.png
  plots/max_efficiency_vs_d_spacing.png
  plots/efficiency_heatmap_d_vs_blaze.png
```

Read the **heatmap first**: the two curves are slices through it, and only the
map shows whether an optimum sits on a broad plateau or a knife-edge. Then use
the curves to pick designs, and drill into `survey/runs/` when a cell looks
wrong -- `theta_scans/theta_scan_9000eV.csv` and its PNG show the actual
incidence-angle scan, and `search_parameters.json` records exactly what was fed
in.

The full 50 x 28 run above puts the optimum at **d = 3.102 nm, blaze 0.859
deg, efficiency 0.69**, with a sharp collapse to near zero for d = 4-6 nm and a
much weaker secondary rise past it -- the step pattern visible in the optimal
blaze curve. A coarse d grid can step straight over a resonance this narrow.

### Why the rough theta-search window has to be wide

The search is seeded from the analytical multilayer Bragg estimate, and for
shallow inside orders that estimate **overshoots the true grating optimum**.
For the survey's best cell (d = 3.102 nm, blaze = 0.859 deg), `survey.csv`
records `bragg_estimate_deg = 1.315` but `incidence_angle_deg = 0.528` -- the
real peak sits **0.79 deg below** the seed. A `rough_scan_half_width_deg` of
0.2 deg would never reach it and every cell would report a spurious efficiency
from the window's edge, which is why the example ships a wide rough window
(`rough_scan_half_width_deg = 1`) narrowed only for the fine pass.

The peak is also narrow: `precise_fwhm_deg = 0.064` for that cell. A wide
window therefore needs enough `rough_scan_points` to land on the peak at all --
61 points over +/-1 deg is a 0.033 deg step, about two samples across the
FWHM. Widening the window without adding points is how a survey silently
misses resonances. `edge_clipped` in `survey.csv` flags cells whose selected
angle landed against a scan boundary; a column of those means the window is
wrong, not the physics.

### Choosing designs and scanning them

```console
$ python 1_run_energy_scan.py                      # optimal blaze at every d
$ python 1_run_energy_scan.py --best               # only the global best cell
$ python 1_run_energy_scan.py --pairs "3.1,0.86; 3.5,0.96"
```

Each design gets `plots/efficiency_vs_energy_Ru-B4C_order2_d<d>nm_blaze<b>deg.png`
in the shared `plots/` folder alongside step 1's headline plots, and two or more
designs also produce
`plots/efficiency_vs_energy_comparison_Ru-B4C_order2.png` with every curve on
one axis -- that overlay is the point of scanning several designs.

### Iterating without re-solving

Both scripts take `--eval`, which re-derives everything from the artifacts
already on disk. After a long survey, `python 0_run_survey.py --eval` rebuilds
`survey.csv` and regenerates all three plots in seconds, so changing how results
are *presented* never costs a re-run. The same applies to
`1_run_energy_scan.py --eval`.

Checkpointing (`CHECKPOINT = True`, `RESUME = True`) means an interrupted run
picks up where it stopped: re-running the same command re-solves only what is
missing.

## Using the API directly

The example scripts are a thin CLI over the same two calls. A smaller,
self-contained version:

```python
from pathlib import Path

from grax import MultilayerDesignConfig, MultilayerGratingDesigner, ThetaSearchScanSettings

config = MultilayerDesignConfig(
    output_dir=Path("examples/simulation/multilayer_grating_design/results"),
    grating_density_lpermm=2400.0,
    diffraction_order=2,
    multilayer_bragg_order=1,
    target_energy_ev=9000.0,
    material_a=("Ru", 12.1),
    material_b=("C", 2.52),            # B4C modelled with the carbon table
    substrate_material=("Si", 2.33),
    n_bilayers=40,
    gamma=0.5,
    d_min_nm=1.5,
    d_max_nm=6.0,
    d_points=4,
    blaze_min_deg=0.6,
    blaze_max_deg=2.0,
    blaze_points=6,
    # Coarser/faster for the 4x6=24-cell survey; the defaults (finer) are kept
    # for the energy scan since it runs far fewer searches.
    survey_scan_settings=ThetaSearchScanSettings(
        rough_scan_points=31, fine_scan_points=41, final_fourier_orders=15
    ),
    energy_scan_min_ev=3000.0,
    energy_scan_max_ev=12000.0,
    energy_scan_points=15,
    solver="neviere",
    polarization="p",
)

designer = MultilayerGratingDesigner(config)
survey = designer.run_survey()
for d_nm, blaze_opt, eff in zip(
    survey.d_values_nm, survey.optimal_blaze_deg, survey.optimal_blaze_efficiency
):
    print(f"d {d_nm:.2f} nm -> optimal blaze {blaze_opt:.2f} deg (eff {eff:.3g})")

# Scan the optimal design for the first d-spacing over energy.
designer.run_energy_scan([(float(survey.d_values_nm[0]), float(survey.optimal_blaze_deg[0]))])
```

## Progress and abort hooks

Both `run_survey` and `run_energy_scan` accept an optional `progress_callback`
(called with a {class}`grax.StageProgress` before each item and once with
`current_label="done"`), and both accept the two stop mechanisms:

- `should_continue` -- a callable checked **between items**. Returning `False`
  stops after the cell or design in flight finishes. `run_survey` then computes
  the optimal blaze per d-spacing from the completed subset and sets
  `SurveyResult.aborted`.
- `stop_event` -- a {class}`threading.Event` forwarded into the theta search
  itself. Setting it stops **immediately**: queued energies are not submitted and
  the live worker processes are terminated. A cell killed this way is discarded
  (a half-written summary CSV would break later `evaluate_survey` calls) and a
  half-scanned design is not returned, but its checkpoint keeps every energy
  already solved, so re-running resumes from there.

Because an in-process solve cannot be interrupted, passing a `stop_event` routes
execution through worker processes even at `max_workers=1`. Callers that pass
none keep the cheaper in-process path.

## Solver selection

`solver` (`rcwa` or `neviere`), `polarization` and `backend` come from the
config and are used by both steps. See {doc}`choosing-a-solver` and
{doc}`multilayer-theta-search`.
