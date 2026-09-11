"""Unit tests for the :mod:`grax.multilayer_design` survey + energy-scan workflow.

The Bragg-angle estimate and the energy-sweep helper
(:func:`grax.run_multilayer_theta_search_sweep`, used by both steps) are replaced
with deterministic fakes so the grid construction, optimal-blaze extraction,
per-run artifact wiring and progress/abort plumbing are exercised without a real
solve. The sweep fake writes a miniature version of the real artifact bundle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from grax import multilayer_design as md
from grax.multilayer_design import (
    MultilayerDesignConfig,
    MultilayerGratingDesigner,
    StageProgress,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class _FakeSweepResult:
    summary_csv_path: Path
    all_orders_csv_path: Path
    energy_efficiency_plot_path: Path
    theta_scan_directory: Path
    profile_plot_path: Path
    stack_plot_path: Path


def _fake_bragg(*, grating: object, energy_ev: float, multilayer_bragg_order: int = 1) -> float:
    """Bragg estimate that shrinks with the bilayer period, like the real one."""

    return float(2.0 / grating.coating_stack.d_period_nm * multilayer_bragg_order + 0.4)


def _fake_sweep(*, grating, energies_ev, output_dir, **_kwargs) -> _FakeSweepResult:
    """Write a miniature theta-search artifact bundle: bump at d=3.0 nm, blaze=1.1 deg."""

    output_dir = Path(output_dir)
    scan_dir = output_dir / "theta_scans"
    scan_dir.mkdir(parents=True, exist_ok=True)
    d_nm = float(grating.coating_stack.d_period_nm)
    blaze = float(grating.blaze_angle_deg)
    theta = 0.5 + 0.05 * d_nm

    summary_rows = []
    for energy in energies_ev:
        efficiency = (
            0.3
            * np.exp(-((d_nm - 3.0) / 0.9) ** 2)
            * np.exp(-((blaze - 1.1) / 0.3) ** 2)
            * np.exp(-((float(energy) - 9000.0) / 6000.0) ** 2)
        )
        summary_rows.append(
            {
                "energy_ev": float(energy),
                "selected_grazing_angle_deg": theta,
                "selected_efficiency": float(efficiency),
                "precise_fwhm_deg": 0.03,
            }
        )
        rough = np.linspace(theta - 0.4, theta + 0.4, 9)
        precise = np.linspace(theta - 0.1, theta + 0.1, 21)
        scan = pd.concat(
            [
                pd.DataFrame(
                    {
                        "scan_type": "rough",
                        "grazing_angle_deg": rough,
                        "selected_efficiency": efficiency
                        * np.exp(-(((rough - theta) / 0.2) ** 2)),
                        "is_selected_peak": 0,
                    }
                ),
                pd.DataFrame(
                    {
                        "scan_type": "precise",
                        "grazing_angle_deg": precise,
                        "selected_efficiency": efficiency
                        * np.exp(-(((precise - theta) / 0.05) ** 2)),
                        "is_selected_peak": (np.isclose(precise, theta)).astype(int),
                    }
                ),
            ],
            ignore_index=True,
        )
        scan.to_csv(scan_dir / f"theta_scan_{int(round(float(energy)))}eV.csv", index=False)

    summary_csv = output_dir / "multilayer_theta_search_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    all_orders_csv = output_dir / "multilayer_theta_search_all_orders.csv"
    all_orders_csv.write_text("energy_ev,order,efficiency\n", encoding="utf-8")
    plot = output_dir / "multilayer_theta_search_energy_vs_efficiency.png"
    plot.write_bytes(b"")
    return _FakeSweepResult(
        summary_csv_path=summary_csv,
        all_orders_csv_path=all_orders_csv,
        energy_efficiency_plot_path=plot,
        theta_scan_directory=scan_dir,
        profile_plot_path=output_dir / "profile.png",
        stack_plot_path=output_dir / "stack.png",
    )


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install the Bragg-estimate and sweep fakes."""

    monkeypatch.setattr(md, "estimate_multilayer_bragg_angle_deg", _fake_bragg)
    monkeypatch.setattr(md, "run_multilayer_theta_search_sweep", _fake_sweep)


def _config(tmp_path: Path, **overrides: object) -> MultilayerDesignConfig:
    base: dict[str, object] = dict(
        output_dir=tmp_path,
        d_min_nm=1.5,
        d_max_nm=6.0,
        d_points=4,
        blaze_min_deg=0.8,
        blaze_max_deg=1.4,
        blaze_points=3,
        energy_scan_min_ev=8000.0,
        energy_scan_max_ev=10000.0,
        energy_scan_points=3,
        show_progress=False,
    )
    base.update(overrides)
    return MultilayerDesignConfig(**base)


# --------------------------------------------------------------------------- #
# Config validation                                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"gamma": 0.0}, "gamma"),
        ({"d_min_nm": 6.0, "d_max_nm": 1.5}, "d grid"),
        ({"d_points": 1}, "d_points"),
        ({"blaze_min_deg": 2.0, "blaze_max_deg": 1.0}, "blaze grid"),
        ({"blaze_points": 0}, "blaze_points"),
        ({"energy_scan_points": 1}, "energy_scan_points"),
        ({"solver": "bogus"}, "solver"),
        ({"on_error": "boom"}, "on_error"),
    ],
)
def test_config_validation(tmp_path: Path, overrides: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _config(tmp_path, **overrides)


def test_grids_are_inclusive(tmp_path: Path) -> None:
    config = _config(tmp_path)
    np.testing.assert_allclose(config.d_grid_nm(), [1.5, 3.0, 4.5, 6.0])
    np.testing.assert_allclose(config.blaze_grid_deg(), [0.8, 1.1, 1.4])
    np.testing.assert_allclose(config.energy_grid_ev(), [8000.0, 9000.0, 10000.0])


# --------------------------------------------------------------------------- #
# Survey                                                                       #
# --------------------------------------------------------------------------- #
def test_run_survey_writes_artifacts_and_optimal_blaze(fakes: None, tmp_path: Path) -> None:
    result = MultilayerGratingDesigner(_config(tmp_path)).run_survey()

    assert result.efficiency_map.shape == (4, 3)
    assert result.combined_csv_path.is_file()
    assert result.plot_path.is_file()
    assert result.efficiency_plot_path.is_file()
    assert result.heatmap_plot_path.is_file()

    # One period folder per d-spacing, each with an overlay + one sub-folder per blaze.
    assert len(result.period_dirs) == 4
    assert [d.name for d in result.period_dirs] == [
        f"d{d:.3f}nm" for d in (1.5, 3.0, 4.5, 6.0)
    ]
    for period_dir in result.period_dirs:
        assert period_dir.parent == result.runs_dir
        assert (period_dir / "overlay.png").is_file()

    # One folder per (d, blaze) run, nested under its period, holding the sweep bundle.
    assert len(result.run_dirs) == 12
    assert all(d.parent.parent == result.runs_dir for d in result.run_dirs)
    assert result.run_dirs[0].name.startswith("blaze")
    for run_dir in result.run_dirs:
        assert (run_dir / "multilayer_theta_search_summary.csv").is_file()
        assert (run_dir / "multilayer_theta_search_all_orders.csv").is_file()
        assert (run_dir / "search_parameters.json").is_file()
        assert list((run_dir / "theta_scans").glob("theta_scan_*.csv"))
    scan = pd.read_csv(next((result.run_dirs[0] / "theta_scans").glob("theta_scan_*.csv")))
    assert set(scan["scan_type"]) == {"rough", "precise"}
    assert scan["is_selected_peak"].sum() >= 1
    record = json.loads((result.run_dirs[0] / "search_parameters.json").read_text())
    assert record["rough_scan_points"] == 61
    assert record["solver"] == "neviere"
    assert "bragg_estimate_deg" in record

    table = pd.read_csv(result.combined_csv_path)
    assert list(table.columns) == [
        "d_nm",
        "blaze_deg",
        "bragg_estimate_deg",
        "incidence_angle_deg",
        "peak_efficiency",
        "precise_fwhm_deg",
        "edge_clipped",
    ]
    assert len(table) == 12

    # The fake efficiency bump peaks at blaze = 1.1 deg for every d-spacing row.
    np.testing.assert_allclose(result.optimal_blaze_deg, [1.1, 1.1, 1.1, 1.1])
    assert not result.aborted
    assert np.all(np.isfinite(result.optimal_blaze_efficiency))


def test_run_survey_progress_is_monotonic_and_complete(fakes: None, tmp_path: Path) -> None:
    reports: list[StageProgress] = []
    MultilayerGratingDesigner(_config(tmp_path)).run_survey(progress_callback=reports.append)

    assert [r.completed for r in reports] == sorted(r.completed for r in reports)
    assert all(r.stage == "survey" and r.total == 12 for r in reports)
    assert reports[-1].current_label == "done"
    assert reports[-1].completed == 12


def test_run_survey_should_continue_stops_early(fakes: None, tmp_path: Path) -> None:
    calls = {"n": 0}

    def stop_after_five() -> bool:
        calls["n"] += 1
        return calls["n"] <= 5

    result = MultilayerGratingDesigner(_config(tmp_path)).run_survey(
        should_continue=stop_after_five
    )

    assert result.aborted
    assert len(result.results) == 5
    # Unreached cells stay NaN in the maps.
    assert np.isnan(result.efficiency_map).sum() == 12 - 5
    # d = 1.5 nm (row full) and d = 3.0 nm (row partial) still get a period folder.
    assert len(result.period_dirs) == 2
    assert all((d / "overlay.png").is_file() for d in result.period_dirs)
    assert np.isfinite(result.optimal_blaze_deg[:2]).all()
    assert np.isnan(result.optimal_blaze_deg[2:]).all()


def test_run_survey_before_first_cell_raises(fakes: None, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="before any result"):
        MultilayerGratingDesigner(_config(tmp_path)).run_survey(should_continue=lambda: False)


def test_run_survey_continue_on_error_records_nan_row(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(md, "estimate_multilayer_bragg_angle_deg", _fake_bragg)

    def _boom_on_large_d(*, grating, energies_ev, output_dir, **kwargs):
        if float(grating.coating_stack.d_period_nm) > 4.0:
            raise RuntimeError("solver blew up")
        return _fake_sweep(grating=grating, energies_ev=energies_ev, output_dir=output_dir, **kwargs)

    monkeypatch.setattr(md, "run_multilayer_theta_search_sweep", _boom_on_large_d)

    result = MultilayerGratingDesigner(_config(tmp_path, on_error="continue")).run_survey()
    table = pd.read_csv(result.combined_csv_path)
    failed = table[table["d_nm"] > 4.0]
    assert len(failed) == 6
    assert failed["peak_efficiency"].isna().all()
    # Only the two d-spacings that solved get a period folder; six run folders.
    assert len(result.period_dirs) == 2
    assert len(result.run_dirs) == 6
    assert all((d / "search_parameters.json").is_file() for d in result.run_dirs)
    assert np.isnan(result.optimal_blaze_deg[2:]).all()


def test_run_survey_fail_fast_propagates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(md, "estimate_multilayer_bragg_angle_deg", _fake_bragg)

    def _always_boom(**_kwargs):
        raise RuntimeError("solver blew up")

    monkeypatch.setattr(md, "run_multilayer_theta_search_sweep", _always_boom)
    with pytest.raises(RuntimeError, match="solver blew up"):
        MultilayerGratingDesigner(_config(tmp_path, on_error="fail_fast")).run_survey()


# --------------------------------------------------------------------------- #
# Energy scan                                                                  #
# --------------------------------------------------------------------------- #
def test_run_energy_scan_fans_out_over_pairs(fakes: None, tmp_path: Path) -> None:
    designer = MultilayerGratingDesigner(_config(tmp_path))
    results = designer.run_energy_scan([(3.0, 1.1), (4.5, 0.9)])

    assert [(round(r.d_spacing_nm, 3), round(r.blaze_angle_deg, 3)) for r in results] == [
        (3.0, 1.1),
        (4.5, 0.9),
    ]
    for scan in results:
        assert scan.summary_csv_path.is_file()
        assert scan.titled_plot_path.is_file()
        assert len(scan.results) == 3
    assert results[0].output_dir != results[1].output_dir
    assert results[0].titled_plot_path != results[1].titled_plot_path


def test_run_energy_scan_requires_pairs(fakes: None, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        MultilayerGratingDesigner(_config(tmp_path)).run_energy_scan([])


def test_coating_label_defaults_to_material_names(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert md._coating_label(config) == "Ru/C"


def test_coating_label_override_shows_real_compound(tmp_path: Path) -> None:
    config = _config(tmp_path, coating_label="Ru/B4C")
    assert md._coating_label(config) == "Ru/B4C"


def test_energy_scan_title_names_coating_and_design(tmp_path: Path) -> None:
    config = _config(tmp_path, coating_label="Ru/B4C", diffraction_order=2)
    title = md._energy_scan_title(config, 3.102, 0.859)
    assert "Ru/B4C" in title
    assert "d = 3.102 nm" in title
    assert "blaze = 0.859 deg" in title
    assert "order 2" in title


# --------------------------------------------------------------------------- #
# Re-evaluate from disk                                                        #
# --------------------------------------------------------------------------- #
def test_evaluate_survey_rebuilds_from_disk_without_solving(
    fakes: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    designer = MultilayerGratingDesigner(_config(tmp_path))
    original = designer.run_survey()
    original.combined_csv_path.unlink()
    original.plot_path.unlink()

    def _boom(**_kwargs):
        raise AssertionError("evaluate_survey must not call the solver")

    monkeypatch.setattr(md, "run_multilayer_theta_search_sweep", _boom)

    result = designer.evaluate_survey()
    assert result.combined_csv_path.is_file()
    assert result.plot_path.is_file()
    assert result.efficiency_plot_path.is_file()
    assert result.heatmap_plot_path.is_file()
    assert len(result.run_dirs) == len(original.run_dirs)
    np.testing.assert_allclose(result.optimal_blaze_deg, original.optimal_blaze_deg)
    np.testing.assert_allclose(result.efficiency_map, original.efficiency_map, equal_nan=True)


def test_evaluate_survey_without_runs_raises(fakes: None, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="no completed survey runs"):
        MultilayerGratingDesigner(_config(tmp_path)).evaluate_survey()


def test_evaluate_energy_scan_reads_existing_designs(
    fakes: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    designer = MultilayerGratingDesigner(_config(tmp_path))
    designer.run_energy_scan([(3.0, 1.1), (4.5, 0.9)])

    def _boom(**_kwargs):
        raise AssertionError("evaluate_energy_scan must not call the solver")

    monkeypatch.setattr(md, "run_multilayer_theta_search_sweep", _boom)

    picked = designer.evaluate_energy_scan([(3.0, 1.1)])
    assert [(round(r.d_spacing_nm, 3), round(r.blaze_angle_deg, 3)) for r in picked] == [(3.0, 1.1)]
    assert picked[0].summary_csv_path.is_file()
    assert picked[0].titled_plot_path.is_file()

    everything = designer.evaluate_energy_scan()
    assert {(round(r.d_spacing_nm, 3), round(r.blaze_angle_deg, 3)) for r in everything} == {
        (3.0, 1.1),
        (4.5, 0.9),
    }


def test_evaluate_energy_scan_without_results_raises(fakes: None, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="no completed energy scans"):
        MultilayerGratingDesigner(_config(tmp_path)).evaluate_energy_scan()
