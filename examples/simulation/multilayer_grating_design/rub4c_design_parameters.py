"""Ru/B4C configuration for the multilayer-grating design example.

Edit the values here, then run ``0_run_survey.py`` followed by
``1_run_energy_scan.py`` (or ``run_all.sh``). The scripts never rewrite this
file.

Workflow:

* ``0_run_survey.py`` scans a 2-D grid of bilayer d-spacing x blaze angle. For
  every pair it builds the multilayer-coated blazed grating and runs graxPy's
  single-energy multilayer theta search at ``target_energy_ev`` -- the search
  scans the incident angle and returns the angle that maximizes the
  selected-order efficiency. It writes ``survey/survey.csv``, three headline
  plots (``plot/optimal_blaze_vs_d_spacing.png``,
  ``plot/max_efficiency_vs_d_spacing.png``, and
  ``plot/efficiency_heatmap_d_vs_blaze.png`` -- the full ``(d, blaze)``
  efficiency map with the optimal-blaze ridge overlaid), and a per-run folder
  tree under ``survey/runs/``: ``d<d>nm/overlay.png`` per period, and
  ``d<d>nm/blaze<b>deg/`` per run with that theta search's full output
  (summary CSV, all-orders CSV, ``theta_scans/``, profile/stack plots,
  checkpoint) plus a ``search_parameters.json``. ``--eval`` re-derives all of
  this from the runs already on disk without solving anything.
* ``1_run_energy_scan.py`` sweeps ``(d, blaze)`` designs from the survey over
  ``[energy_scan_min_ev, energy_scan_max_ev]``: the optimal blaze per d-spacing
  by default, only the single best with ``--best``, or exactly ``--pairs
  "d,blaze; ..."``. ``--eval`` re-reads existing ``energy_scan/`` results
  instead of re-solving.

There is no CFF input: the incident angle is a result of the theta search for
every pair. ``gamma`` is held fixed; the anti-blaze angle defaults to 0 (a
plain sawtooth).

The config below is split into three sections, marked ``# ===`` below, matching
:class:`grax.MultilayerDesignConfig`'s own grouping -- check that comment before
moving a value, since editing the wrong section is a common way to get "I
changed X but nothing happened" (e.g. ``rough_scan_points`` inside
``SURVEY_SCAN`` only affects ``0_run_survey.py``; ``1_run_energy_scan.py`` reads
``ENERGY_SCAN_SCAN`` instead):

1. ``SHARED`` -- grating geometry, materials, solver/runtime. Used by both
   scripts.
2. ``SURVEY`` -- read only by ``0_run_survey.py`` (grid, ``SURVEY_SCAN``
   settings, ``on_error``).
3. ``ENERGY_SCAN`` -- read only by ``1_run_energy_scan.py`` (energy range,
   ``ENERGY_SCAN_SCAN`` settings, worker/progress/tracking controls).

For a fast smoke run: shrink ``d_points``/``blaze_points`` to 2 in ``SURVEY``,
``rough_scan_points``/``fine_scan_points`` to ~15 and the Fourier orders to
3/5/7 in *both* ``SURVEY_SCAN`` and ``ENERGY_SCAN_SCAN``, the rough/fine/final
resolutions to 2.0/1.0/1.0 nm, and ``energy_scan_points`` to 3 in
``ENERGY_SCAN``.
"""

from __future__ import annotations

from pathlib import Path

from grax import MultilayerDesignConfig, ThetaSearchScanSettings

# ========================================================================== #
# SHARED -- grating geometry, materials, solver/runtime.                     #
# Read by both 0_run_survey.py and 1_run_energy_scan.py.                     #
# ========================================================================== #
OUTPUT_DIR = Path(__file__).resolve().parent / "results"

GRATING_DENSITY_LPERMM = 2400.0
DIFFRACTION_ORDER = 2
MULTILAYER_BRAGG_ORDER = 1

# Ru/B4C on a silicon substrate; B4C is modelled with the carbon table, so
# COATING_LABEL overrides what plot titles call it -- otherwise they would say
# "Ru/C", the table's element name, not the real compound.
MATERIAL_A = ("Ru", 12.1)
MATERIAL_B = ("C", 2.52)
COATING_LABEL = "Ru/B4C"
SUBSTRATE_MATERIAL = ("Si", 2.33)
N_BILAYERS = 40
GAMMA = 0.5
ANTI_BLAZE_ANGLE_DEG = 0.0

X_RESOLUTION_NM = 0.5
Z_RESOLUTION_NM = 0.5

BACKEND = "numba"
SOLVER = "neviere"
POLARIZATION = "p"
CHECKPOINT = True
RESUME = True
SAVE_PROFILE_PLOT = True
SAVE_STACK_PLOT = True

# ========================================================================== #
# SURVEY -- read only by 0_run_survey.py (run_survey / evaluate_survey).     #
# ========================================================================== #
TARGET_ENERGY_EV = 9000.0

# d-spacing x blaze-angle grid.
D_MIN_NM = 2.0
D_MAX_NM = 8.0
D_POINTS = 50  # 10
BLAZE_MIN_DEG = 0.6
BLAZE_MAX_DEG = 2.0
BLAZE_POINTS = 28

# Theta-search scan settings for the survey: one single-energy search per
# (d, blaze) grid cell (many cheap searches -- 1400 with the grid above), so
# this is usually the one to keep fast. The rough half-width is wide because
# the scan is seeded from the multilayer Bragg estimate, which overshoots the
# true grating optimum for shallow inside orders; the window has to reach
# well below it across the whole d grid.
SURVEY_SCAN = ThetaSearchScanSettings(
    rough_scan_half_width_deg=1,
    rough_scan_points=61,
    rough_fourier_orders=5,
    rough_x_resolution_nm=1.0,
    rough_z_resolution_nm=1.0,
    fine_scan_half_width_deg=0.2,
    fine_scan_points=81,
    fine_fourier_orders=15,
    fine_x_resolution_nm=0.5,
    fine_z_resolution_nm=0.5,
    final_fourier_orders=25,
    final_x_resolution_nm=0.2,
    final_z_resolution_nm=0.2,
    precise_peak_selection_mode="max",
)

ON_ERROR = "continue"

# ========================================================================== #
# ENERGY_SCAN -- read only by 1_run_energy_scan.py                           #
# (run_energy_scan / evaluate_energy_scan).                                  #
# ========================================================================== #
ENERGY_SCAN_MIN_EV = 3000.0
ENERGY_SCAN_MAX_EV = 12000.0
ENERGY_SCAN_POINTS = 1000

# Theta-search scan settings for the energy scan: one search per energy, for
# each chosen (d, blaze) design -- few designs, but often many energies each,
# so this is usually the one to make more precise.
ENERGY_SCAN_SCAN = ThetaSearchScanSettings(
    rough_scan_half_width_deg=1,
    rough_scan_points=61,
    rough_fourier_orders=5,
    rough_x_resolution_nm=1.0,
    rough_z_resolution_nm=1.0,
    fine_scan_half_width_deg=0.2,
    fine_scan_points=81,
    fine_fourier_orders=15,
    fine_x_resolution_nm=0.5,
    fine_z_resolution_nm=0.5,
    final_fourier_orders=25,
    final_x_resolution_nm=0.2,
    final_z_resolution_nm=0.2,
    precise_peak_selection_mode="max",
)

MAX_WORKERS = "auto"
SHOW_PROGRESS = True
THETA_TRACKING_MODE = "auto"
MAX_TRACKING_ENERGY_STEP_EV = None

CONFIG = MultilayerDesignConfig(
    output_dir=OUTPUT_DIR,
    # -- Shared --
    grating_density_lpermm=GRATING_DENSITY_LPERMM,
    diffraction_order=DIFFRACTION_ORDER,
    multilayer_bragg_order=MULTILAYER_BRAGG_ORDER,
    material_a=MATERIAL_A,
    material_b=MATERIAL_B,
    coating_label=COATING_LABEL,
    substrate_material=SUBSTRATE_MATERIAL,
    n_bilayers=N_BILAYERS,
    gamma=GAMMA,
    anti_blaze_angle_deg=ANTI_BLAZE_ANGLE_DEG,
    x_resolution_nm=X_RESOLUTION_NM,
    z_resolution_nm=Z_RESOLUTION_NM,
    backend=BACKEND,
    solver=SOLVER,
    polarization=POLARIZATION,
    checkpoint=CHECKPOINT,
    resume=RESUME,
    save_profile_plot=SAVE_PROFILE_PLOT,
    save_stack_plot=SAVE_STACK_PLOT,
    # -- Survey only --
    target_energy_ev=TARGET_ENERGY_EV,
    d_min_nm=D_MIN_NM,
    d_max_nm=D_MAX_NM,
    d_points=D_POINTS,
    blaze_min_deg=BLAZE_MIN_DEG,
    blaze_max_deg=BLAZE_MAX_DEG,
    blaze_points=BLAZE_POINTS,
    survey_scan_settings=SURVEY_SCAN,
    on_error=ON_ERROR,
    # -- Energy scan only --
    energy_scan_min_ev=ENERGY_SCAN_MIN_EV,
    energy_scan_max_ev=ENERGY_SCAN_MAX_EV,
    energy_scan_points=ENERGY_SCAN_POINTS,
    energy_scan_settings=ENERGY_SCAN_SCAN,
    max_workers=MAX_WORKERS,
    show_progress=SHOW_PROGRESS,
    theta_tracking_mode=THETA_TRACKING_MODE,
    max_tracking_energy_step_ev=MAX_TRACKING_ENERGY_STEP_EV,
)
