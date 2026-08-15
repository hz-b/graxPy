from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import grax
from grax import AFMGrating, AFMPreprocessing, BatchSimulationRunner, run_simulation
from grax.gratings import ProfileGrating
from tests.optical_constants import load_optical_constants_table

OPTICAL_CONSTANTS_DIR = Path(__file__).resolve().parents[2] / "examples" / "optical_constants"
EXAMPLES_GRATING_DIR = Path(__file__).resolve().parents[2] / "examples" / "grating"
SI = load_optical_constants_table(OPTICAL_CONSTANTS_DIR / "n_Si_cxro.txt", "Si")
PT = load_optical_constants_table(OPTICAL_CONSTANTS_DIR / "n_Pt_cxro.txt", "Pt")


def _synthetic_afm_data(
    *,
    n_periods: int = 6,
    period_nm: float = 1600.0,
    samples_per_period: int = 80,
) -> np.ndarray:
    x_nm = np.linspace(0.0, n_periods * period_nm, n_periods * samples_per_period + 1)
    z_nm = 10.0 - 5.0 * np.cos(2.0 * np.pi * x_nm / period_nm)
    return np.column_stack((x_nm, z_nm))


def _synthetic_laminar_afm_with_secondary_minima(
    *,
    n_periods: int = 6,
    period_nm: float = 1600.0,
    samples_per_period: int = 800,
) -> np.ndarray:
    """Return a laminar-like scan with shallow secondary minima."""

    x_nm = np.linspace(0.0, n_periods * period_nm, n_periods * samples_per_period + 1)
    z_nm = (
        5.0
        - 2.0 * np.cos(2.0 * np.pi * x_nm / period_nm)
        + 0.8 * np.cos(4.0 * np.pi * x_nm / period_nm + 0.1)
        + 0.1 * np.cos(6.0 * np.pi * x_nm / period_nm)
    )
    return np.column_stack((x_nm, z_nm))


def _synthetic_laminar_wall_profile(
    *,
    n_periods: int = 6,
    period_nm: float = 1600.0,
    samples_per_period: int = 400,
    valley_start_fraction: float = 0.1,
    valley_end_fraction: float = 0.85,
    transition_fraction: float = 0.012,
) -> np.ndarray:
    """Return a laminar-like scan with steep walls and a broad valley floor."""

    x_nm = np.linspace(0.0, n_periods * period_nm, n_periods * samples_per_period + 1)
    phase = (x_nm % period_nm) / period_nm
    transition = transition_fraction
    left_step = 0.5 * (1.0 + np.tanh((phase - valley_start_fraction) / transition))
    right_step = 0.5 * (1.0 + np.tanh((phase - valley_end_fraction) / transition))
    valley_mask = np.clip(left_step - right_step, 0.0, 1.0)
    z_nm = 15.0 - 14.0 * valley_mask
    return np.column_stack((x_nm, z_nm))


def _build_processed_afm(period_nm: float = 1600.0) -> AFMPreprocessing:
    afm = AFMPreprocessing(
        _synthetic_afm_data(period_nm=period_nm),
        units="nm",
        save_plots=False,
        show_plots=False,
    )
    afm.normalize_scan(reverse=False, zero_baseline=True)
    afm.find_troughs(period_nm=period_nm, min_separation_fraction=0.4)
    afm.extract_period(average=True)
    afm.rescale_period(period_nm=period_nm)
    return afm


def test_afm_preprocessing_converts_units_to_nanometers() -> None:
    data_nm = _synthetic_afm_data(n_periods=2, period_nm=1000.0)
    data_m = data_nm * 1e-9
    afm = AFMPreprocessing(data_m, units="m", save_plots=False, show_plots=False)
    assert afm.x_nm[1] == pytest.approx(data_nm[1, 0])
    assert afm.z_nm[3] == pytest.approx(data_nm[3, 1])


def test_afm_preprocessing_requires_find_troughs_before_extract() -> None:
    afm = AFMPreprocessing(_synthetic_afm_data(), units="nm", save_plots=False, show_plots=False)
    with pytest.raises(RuntimeError, match="find_troughs"):
        afm.extract_period()


def test_afm_preprocessing_single_and_average_period_extraction() -> None:
    period_nm = 1600.0
    afm = AFMPreprocessing(
        _synthetic_afm_data(period_nm=period_nm),
        units="nm",
        save_plots=False,
        show_plots=False,
    )
    afm.normalize_scan(zero_baseline=True)
    afm.find_troughs(period_nm=period_nm, min_separation_fraction=0.4)
    afm.extract_period(period_index=1, average=False)
    assert afm.period_x is not None
    assert afm.period_z is not None
    assert afm.period_x[0] == pytest.approx(0.0)
    assert afm.period_x[-1] == pytest.approx(1.0)

    afm.extract_period(average=True)
    afm.rescale_period(period_nm=period_nm)
    x_nm, z_nm = afm.get_profile()
    assert x_nm[0] == pytest.approx(0.0)
    assert x_nm[-1] == pytest.approx(period_nm)
    assert z_nm.shape == x_nm.shape


def test_afm_preprocessing_periodicity_ramp_matches_endpoints() -> None:
    afm = AFMPreprocessing(_synthetic_afm_data(), units="nm", save_plots=False, show_plots=False)
    afm.normalize_scan(zero_baseline=True)
    afm.find_troughs(period_nm=1600.0, min_separation_fraction=0.4)
    afm.extract_period(period_index=0, average=False)
    assert afm.period_z is not None
    afm.period_z[-1] += 2.5
    afm.apply_periodicity_ramp()
    assert afm.period_z is not None
    assert afm.period_z[0] == pytest.approx(afm.period_z[-1], abs=1e-12)


def test_afm_find_troughs_filters_shallow_secondary_laminar_minima() -> None:
    period_nm = 1600.0
    afm = AFMPreprocessing(
        _synthetic_laminar_afm_with_secondary_minima(period_nm=period_nm),
        units="nm",
        save_plots=False,
        show_plots=False,
    )
    afm.normalize_scan(zero_baseline=True)

    afm.find_troughs(
        period_nm=period_nm,
        min_separation_fraction=0.4,
        min_prominence_fraction=0.0,
    )
    assert afm.trough_indices is not None
    unfiltered_count = len(afm.trough_indices)

    afm.find_troughs(
        period_nm=period_nm,
        min_separation_fraction=0.4,
        min_prominence_fraction=0.1,
    )
    assert afm.trough_indices is not None
    filtered_count = len(afm.trough_indices)

    assert unfiltered_count > filtered_count
    assert filtered_count == 6

    afm.extract_period(average=True)
    afm.rescale_period(period_nm=period_nm)
    x_nm, z_nm = afm.get_profile()
    assert x_nm[0] == pytest.approx(0.0)
    assert x_nm[-1] == pytest.approx(period_nm)
    assert np.all(np.isfinite(z_nm))


def test_afm_find_troughs_keeps_blaze_like_scan_count_with_default_prominence() -> None:
    afm = AFMPreprocessing(_synthetic_afm_data(), units="nm", save_plots=False, show_plots=False)
    afm.normalize_scan(zero_baseline=True)

    afm.find_troughs(period_nm=1600.0, min_separation_fraction=0.4, min_prominence_fraction=0.0)
    assert afm.trough_indices is not None
    unfiltered_count = len(afm.trough_indices)

    afm.find_troughs(period_nm=1600.0, min_separation_fraction=0.4)
    assert afm.trough_indices is not None
    filtered_count = len(afm.trough_indices)

    assert unfiltered_count == filtered_count
    assert filtered_count >= 2


def test_afm_find_troughs_laminar_mode_uses_wall_midpoints() -> None:
    period_nm = 1600.0
    afm = AFMPreprocessing(
        _synthetic_laminar_wall_profile(period_nm=period_nm),
        units="nm",
        save_plots=False,
        show_plots=False,
    )
    afm.normalize_scan(zero_baseline=True)
    afm.find_troughs(period_nm=period_nm, profile_type="laminar")

    assert afm.trough_indices is not None
    trough_x_nm = afm.x_nm[afm.trough_indices]
    expected_centers_nm = ((0.1 + 0.85) * 0.5 + np.arange(6)) * period_nm
    assert trough_x_nm.shape == expected_centers_nm.shape
    assert np.allclose(trough_x_nm, expected_centers_nm, atol=0.08 * period_nm)


def test_afm_find_troughs_laminar_mode_rejects_wall_edge_minima_on_real_data() -> None:
    period_nm = 1e6 / 600.0
    afm_data = np.loadtxt(EXAMPLES_GRATING_DIR / "data" / "afm_profile_example_laminar.txt")
    afm = AFMPreprocessing(afm_data, units="m", save_plots=False, show_plots=False)
    afm.normalize_scan(reverse=True, zero_baseline=True)
    afm.find_troughs(period_nm=period_nm, min_separation_fraction=0.4, profile_type="laminar")

    assert afm.trough_indices is not None
    trough_x_um = afm.x_nm[afm.trough_indices] * 1e-3
    expected_centers_um = np.array([0.85, 3.28, 5.85, 8.36, 10.86, 13.35])
    assert len(trough_x_um) == len(expected_centers_um)
    assert np.allclose(trough_x_um, expected_centers_um, atol=0.25)

    afm.extract_period(average=True)
    afm.rescale_period(period_nm=period_nm)
    x_nm, z_nm = afm.get_profile()
    assert x_nm[0] == pytest.approx(0.0)
    assert x_nm[-1] == pytest.approx(period_nm)
    assert np.all(np.isfinite(z_nm))


def test_profile_grating_depth_uses_explicit_points() -> None:
    grating = ProfileGrating(
        period_lpermm=625,
        x_points_nm=np.array([0.0, 400.0, 800.0, 1200.0, 1600.0]),
        z_points_nm=np.array([0.0, 20.0, 5.0, 18.0, 0.0]),
    )
    x_points, z_points = grating.profile_points()
    assert x_points.size == 5
    assert z_points.size == 5
    assert grating.profile_depth_nm() == pytest.approx(20.0)


def test_afm_symbols_are_exposed_in_public_api() -> None:
    assert "AFMPreprocessing" in grax.__all__
    assert "AFMGrating" in grax.__all__
    assert "ProfileGrating" in grax.__all__


def test_afm_grating_from_preprocessing_runs_single_simulation() -> None:
    period_nm = 1600.0
    afm = _build_processed_afm(period_nm=period_nm)
    grating = AFMGrating.from_preprocessing(
        afm,
        period_lpermm=int(round(1e6 / period_nm)),
        substrate_material=SI,
        layer_material=PT,
        layer_thickness_nm=25.0,
        x_resolution_nm=4.0,
        z_resolution_nm=2.0,
    )
    result = run_simulation(
        grating=grating,
        energy_ev=500.0,
        grazing_angle_deg=4.0,
        diffraction_order=1,
        fourier_orders=3,
    )
    assert result.selected_efficiency >= 0.0


def test_afm_preprocessing_accepts_numpy_array_input() -> None:
    afm = AFMPreprocessing(np.asarray(_synthetic_afm_data(), dtype=float), units="nm", save_plots=False, show_plots=False)
    assert afm.x_nm.ndim == 1
    assert afm.z_nm.ndim == 1


def test_afm_grating_with_string_materials_runs_simulation() -> None:
    afm = _build_processed_afm(period_nm=1600.0)
    grating = AFMGrating.from_preprocessing(
        afm,
        substrate_material="Si",
        layer_material="Pt",
        layer_thickness_nm=25.0,
        x_resolution_nm=4.0,
        z_resolution_nm=2.0,
    )

    result = run_simulation(
        grating=grating,
        energy_ev=500.0,
        grazing_angle_deg=4.0,
        diffraction_order=1,
        fourier_orders=3,
    )

    assert result.selected_efficiency >= 0.0


def test_afm_grating_runs_in_batch_case_path() -> None:
    period_nm = 1600.0
    afm = _build_processed_afm(period_nm=period_nm)
    grating = AFMGrating.from_preprocessing(
        afm,
        period_lpermm=int(round(1e6 / period_nm)),
        substrate_material=SI,
        layer_material=PT,
        layer_thickness_nm=25.0,
        x_resolution_nm=4.0,
        z_resolution_nm=2.0,
    )
    runner = BatchSimulationRunner(fourier_orders=3, show_progress=False)
    cases = [{"grating": grating, "energy_ev": 500.0, "grazing_angle_deg": 4.0}]
    results = list(runner.run_cases(cases))
    assert len(results) == 1
    assert results[0].status == "ok"


def test_afm_preprocessing_saves_plots_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    afm = AFMPreprocessing(_synthetic_afm_data(), units="nm", show_plots=False)
    afm.normalize_scan(zero_baseline=True)

    expected = tmp_path / "results" / "afm_preprocessing" / "01_normalize_scan.png"
    assert expected.exists()


def test_afm_preprocessing_can_disable_plot_saving(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    afm = AFMPreprocessing(_synthetic_afm_data(), units="nm", save_plots=False, show_plots=False)
    afm.normalize_scan(zero_baseline=True)

    expected_dir = tmp_path / "results" / "afm_preprocessing"
    assert not expected_dir.exists()


def test_afm_preprocessing_respects_custom_results_folder(tmp_path: Path) -> None:
    results_dir = tmp_path / "results" / "afm_preprocessing_laminar"
    afm = AFMPreprocessing(
        _synthetic_afm_data(),
        units="nm",
        results_folder=results_dir,
        show_plots=False,
    )
    afm.normalize_scan(zero_baseline=True)
    expected = results_dir / "01_normalize_scan.png"
    assert expected.exists()


def test_afm_tutorial_references_both_example_workflows_and_image_sets() -> None:
    tutorial_path = Path(__file__).resolve().parents[2] / "docs" / "tutorials" / "gratings" / "afm-preprocessing-profile.md"
    build_docs_path = Path(__file__).resolve().parents[2] / "tools" / "build_docs.sh"

    tutorial = tutorial_path.read_text(encoding="utf-8")
    build_docs = build_docs_path.read_text(encoding="utf-8")

    assert "afm_preprocessing_blazed_profile.py" in tutorial
    assert "afm_preprocessing_laminar_profile.py" in tutorial
    assert "afm_preprocessing_blazed/01_normalize_scan.png" in tutorial
    assert "afm_preprocessing_laminar/01_normalize_scan.png" in tutorial
    assert "examples/grating/results/afm_preprocessing_blazed/01_normalize_scan.png" in build_docs
    assert "examples/grating/results/afm_preprocessing_laminar/01_normalize_scan.png" in build_docs


def test_afm_grating_from_preprocessing_infers_period_lpermm() -> None:
    period_nm = 1600.0
    afm = _build_processed_afm(period_nm=period_nm)
    grating = AFMGrating.from_preprocessing(
        afm,
        substrate_material=SI,
        layer_material=PT,
        layer_thickness_nm=25.0,
        x_resolution_nm=4.0,
        z_resolution_nm=2.0,
    )
    assert grating.period_lpermm == int(round(1e6 / period_nm))


def test_afm_grating_from_preprocessing_explicit_period_overrides_inference() -> None:
    period_nm = 1600.0
    afm = _build_processed_afm(period_nm=period_nm)
    grating = AFMGrating.from_preprocessing(
        afm,
        period_lpermm=700,
        substrate_material=SI,
        layer_material=PT,
        layer_thickness_nm=25.0,
        x_resolution_nm=4.0,
        z_resolution_nm=2.0,
    )
    assert grating.period_lpermm == 700


def test_afm_grating_from_preprocessing_missing_rescale_raises_clear_error() -> None:
    afm = AFMPreprocessing(_synthetic_afm_data(), units="nm", save_plots=False, show_plots=False)
    afm.normalize_scan(zero_baseline=True)
    afm.find_troughs(period_nm=1600.0, min_separation_fraction=0.4)
    afm.extract_period(average=True)

    with pytest.raises(RuntimeError, match="rescale_period\\(\\).*period_lpermm explicitly"):
        AFMGrating.from_preprocessing(
            afm,
            substrate_material=SI,
            layer_material=PT,
            layer_thickness_nm=25.0,
            x_resolution_nm=4.0,
            z_resolution_nm=2.0,
        )
