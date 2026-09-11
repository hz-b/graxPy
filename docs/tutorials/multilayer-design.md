# Multilayer grating design workflow

Use {class}`grax.MultilayerGratingDesigner` to size a periodic multilayer coating
for a blazed grating working in a chosen diffraction order. The workflow has two
steps that share one {class}`grax.MultilayerDesignConfig`.

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

There is **no CFF input**. The incident angle is a result of the search for every
pair; the operating fixed-focus behaviour re-emerges later as `theta*(E)` from
step 2.

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
  structure the two curves above are slices of.
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
and `theta*(E)`. Adding `--eval`
({meth}`grax.MultilayerGratingDesigner.evaluate_energy_scan`) re-reads existing
`energy_scan/` results instead of re-solving.

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
far.

The coating name comes from `MultilayerDesignConfig.coating_label` when set,
otherwise from `"<material_a name>/<material_b name>"` -- set it explicitly
whenever a material is modelled with a stand-in optical-constants table (B4C
modelled with the carbon table, in the Ru/B4C example) so the title and
filename show the real compound.

## Example

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

See `examples/simulation/multilayer_grating_design/` for the full runnable
workflow: `rub4c_design_parameters.py` builds the config, `0_run_survey.py` runs
the survey and `1_run_energy_scan.py` scans the optimal design per d-spacing
(`--best` for the single best, `--pairs "d,blaze; ..."` for explicit ones).
`run_all.sh` runs both in order.

## Progress and abort hooks

Both `run_survey` and `run_energy_scan` accept an optional `progress_callback`
(called with a {class}`grax.StageProgress` before each item and once with
`current_label="done"`). `run_survey` also accepts `should_continue`: returning
`False` stops the scan between cells and computes the optimal blaze per d-spacing
from the completed subset, with `SurveyResult.aborted` set.

## Solver selection

`solver` (`rcwa` or `neviere`), `polarization` and `backend` come from the
config and are used by both steps. See {doc}`choosing-a-solver` and
{doc}`multilayer-theta-search`.
