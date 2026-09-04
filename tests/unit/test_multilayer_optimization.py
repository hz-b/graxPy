"""Unit tests for the three-stage :mod:`grax.multilayer_optimization` workflow.

The XRT reflectivity engine and the graxPy theta-search sweep are both replaced
with deterministic fakes so the stage orchestration, the state-file hand-off and
the geometry maths are exercised without a real solve.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import grax
from grax import multilayer_optimization as mo
from grax.multilayer_optimization import (
    MultilayerOptimizationConfig,
    d_spacing_bounds_from_bragg_angles,
    ensure_target_energy,
    intersect_search_bounds,
    resolve_configured_value,
    run_blaze_study,
    run_d_spacing_study,
    run_gamma_study,
    select_target_energy_optimum,
)
from grax.multilayer_reflectivity import _COLUMNS

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #
class _FakeReflectivity:
    """Deterministic stand-in for :class:`grax.MultilayerReflectivity`.

    ``peak_rp`` is a smooth bump centered on ``d = 2.8 nm`` and ``gamma = 0.5``
    at the target energy; ``peak_rs`` is a fixed multiple above it.
    """

    def __init__(
        self,
        material_a: object,
        thickness_a_nm: float,
        material_b: object,
        thickness_b_nm: float,
        n_bilayers: int,
        *,
        save_recap: Path | str | None = None,
        individuals: bool = False,
    ) -> None:
        self.d_nm = float(thickness_a_nm) + float(thickness_b_nm)
        self.gamma = float(thickness_a_nm) / self.d_nm
        self.save_recap = None if save_recap is None else Path(save_recap)

    def reflectivity_vs_energy(self, energies_ev: object, **_kwargs: object) -> pd.DataFrame:
        """Return a synthetic reflectivity table over ``energies_ev``."""

        rows = []
        for energy in (float(value) for value in energies_ev):
            shape = (
                np.exp(-((self.d_nm - 2.8) / 0.6) ** 2)
                * np.exp(-((self.gamma - 0.5) / 0.25) ** 2)
                * np.exp(-((energy - 9000.0) / 6000.0) ** 2)
            )
            peak_rp = 0.1 + 0.7 * shape
            rows.append(
                {
                    "energy_ev": energy,
                    "peak_rs": 1.3 * peak_rp,
                    "peak_rp": peak_rp,
                    "peak_angle_deg": 1.05,
                    "bragg_angle_deg": 1.0,
                    "scan_min_angle_deg": 0.5,
                    "scan_max_angle_deg": 1.5,
                    "fwhm_deg": 0.03,
                    "fwhm_ev": 5.0,
                    "left_half_max_angle_deg": 1.035,
                    "right_half_max_angle_deg": 1.065,
                }
            )
        frame = pd.DataFrame(rows, columns=list(_COLUMNS))
        if self.save_recap is not None:
            self.save_recap.mkdir(parents=True, exist_ok=True)
            frame.to_csv(self.save_recap / "results.csv", index=False)
        return frame


class _FakeSweepResult:
    def __init__(self, summary_csv_path: Path) -> None:
        self.summary_csv_path = summary_csv_path


def _fake_sweep(*, grating: object, energies_ev: object, output_dir: Path, **_kwargs: object):
    """Stand-in for ``run_multilayer_theta_search_sweep``.

    ``selected_efficiency`` peaks at ``blaze_angle_deg == 1.1`` and at the target
    energy, and a real summary CSV is written so the workflow can read it back.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    blaze = float(grating.blaze_angle_deg)
    rows = []
    for energy in (float(value) for value in energies_ev):
        efficiency = (
            0.3
            * np.exp(-((blaze - 1.1) / 0.15) ** 2)
            * np.exp(-((energy - 9000.0) / 6000.0) ** 2)
        )
        rows.append({"energy_ev": energy, "selected_efficiency": efficiency})
    summary_path = output_dir / "multilayer_theta_search_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    return _FakeSweepResult(summary_path)


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install the reflectivity and sweep fakes on the workflow module."""

    monkeypatch.setattr(mo, "MultilayerReflectivity", _FakeReflectivity)
    monkeypatch.setattr(mo, "run_multilayer_theta_search_sweep", _fake_sweep)


def _config(tmp_path: Path, **overrides: object) -> MultilayerOptimizationConfig:
    """Return a small config with tight energy and parameter grids."""

    base: dict[str, object] = {
        "output_dir": tmp_path,
        "d_spacing_points": 5,
        "d_spacing_energy_min_ev": 8800.0,
        "d_spacing_energy_max_ev": 9200.0,
        "d_spacing_energy_step_ev": 200.0,
        "gamma_energy_min_ev": 8800.0,
        "gamma_energy_max_ev": 9200.0,
        "gamma_energy_step_ev": 200.0,
        "gamma_min": 0.4,
        "gamma_max": 0.6,
        "gamma_step": 0.1,
        "blaze_energy_min_ev": 8800.0,
        "blaze_energy_max_ev": 9200.0,
        "blaze_energy_points": 3,
        "blaze_angle_points": 3,
        "blaze_angle_half_range_deg": 0.2,
    }
    base.update(overrides)
    return MultilayerOptimizationConfig(**base)


# --------------------------------------------------------------------------- #
# Pure helpers                                                                 #
# --------------------------------------------------------------------------- #
def test_d_spacing_bounds_from_bragg_angles_inverts_angle_and_d() -> None:
    """The larger grazing angle maps to the smaller d-spacing."""

    d_min, d_max = d_spacing_bounds_from_bragg_angles(9000.0, 0.5, 2.0, bragg_order=1)
    assert 0.0 < d_min < d_max
    wavelength_nm = mo.HC_EV_NM / 9000.0
    assert d_min == pytest.approx(wavelength_nm / (2.0 * np.sin(np.deg2rad(2.0))))
    assert d_max == pytest.approx(wavelength_nm / (2.0 * np.sin(np.deg2rad(0.5))))


def test_intersect_search_bounds_rejects_empty_intersection() -> None:
    """Disjoint derived and practical intervals raise."""

    with pytest.raises(ValueError, match="No usable d-spacing interval"):
        intersect_search_bounds(2.0, 3.0, 5.0, 6.0)


def test_ensure_target_energy_snaps_and_appends() -> None:
    """A near-target sample snaps exactly; otherwise the target is appended."""

    snapped = ensure_target_energy([8999.9999999999, 9500.0], 9000.0)
    assert 9000.0 in snapped
    appended = ensure_target_energy([8000.0, 10000.0], 9000.0)
    assert list(appended) == [8000.0, 9000.0, 10000.0]


def test_select_target_energy_optimum_picks_argmax_at_target() -> None:
    """The best parameter is read off the target-energy rows only."""

    frame = pd.DataFrame(
        {
            "gamma": [0.4, 0.5, 0.4, 0.5],
            "peak_rp": [0.1, 0.2, 0.9, 0.3],
            "energy_ev": [9000.0, 9000.0, 8000.0, 8000.0],
        }
    )
    value, metric = select_target_energy_optimum(
        frame, parameter_column="gamma", metric_column="peak_rp",
        target_energy_ev=9000.0, energy_column="energy_ev",
    )
    assert value == 0.5
    assert metric == 0.2


def test_rounded_d_grid_forces_geometry_value_and_rounds() -> None:
    """The grid is 0.1 nm-rounded and contains the geometry value exactly."""

    grid = mo._rounded_d_grid(2.31, 3.29, points=5, required_nm=2.7)
    assert np.all(np.isclose(grid, np.round(grid, 1)))
    assert 2.7 in grid
    assert grid.min() >= 2.4 and grid.max() <= 3.2


def test_rounded_d_grid_rejects_geometry_value_outside_interval() -> None:
    """A geometry value outside the rounded interval is a hard error."""

    with pytest.raises(ValueError, match="cannot contain"):
        mo._rounded_d_grid(2.5, 3.0, points=5, required_nm=4.0)


def test_stage_energy_grid_quick_switch(tmp_path: Path) -> None:
    """``quick`` swaps the dense step for the quick step."""

    dense = _config(tmp_path)
    quick = _config(tmp_path, quick=True, d_spacing_energy_quick_step_ev=400.0)
    assert len(mo._stage_energy_grid(dense, "d_spacing")) == 3
    assert list(mo._stage_energy_grid(quick, "d_spacing")) == [8800.0, 9000.0, 9200.0]


# --------------------------------------------------------------------------- #
# resolve_configured_value                                                     #
# --------------------------------------------------------------------------- #
def test_resolve_configured_value_numeric_ignores_state(tmp_path: Path) -> None:
    """A numeric value is returned verbatim, state file untouched."""

    state = tmp_path / "state.json"
    state.write_text(json.dumps({"d_suggested_nm": 9.9}), encoding="utf-8")
    assert resolve_configured_value(
        2.7, state_path=state, suggestion_key="d_suggested_nm", parameter_name="d_spacing_nm"
    ) == pytest.approx(2.7)


def test_resolve_configured_value_auto_reads_state(tmp_path: Path) -> None:
    """``"auto"`` resolves the suggestion key from the state file."""

    state = tmp_path / "state.json"
    state.write_text(json.dumps({"d_suggested_nm": 2.9}), encoding="utf-8")
    assert resolve_configured_value(
        "auto", state_path=state, suggestion_key="d_suggested_nm", parameter_name="d_spacing_nm"
    ) == pytest.approx(2.9)


def test_resolve_configured_value_auto_without_state_raises(tmp_path: Path) -> None:
    """``"auto"`` with no state file names the missing prerequisite."""

    with pytest.raises(ValueError, match="does not exist"):
        resolve_configured_value(
            "auto",
            state_path=tmp_path / "missing.json",
            suggestion_key="d_suggested_nm",
            parameter_name="d_spacing_nm",
        )


def test_resolve_configured_value_rejects_non_auto_string(tmp_path: Path) -> None:
    """Any string other than ``"auto"`` is rejected."""

    with pytest.raises(ValueError, match="numeric or 'auto'"):
        resolve_configured_value(
            "best", state_path=tmp_path / "s.json",
            suggestion_key="d_suggested_nm", parameter_name="d_spacing_nm",
        )


# --------------------------------------------------------------------------- #
# Config validation                                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"gamma": 1.5}, "0 < gamma < 1"),
        ({"solver": "octave"}, "solver must be"),
        ({"d_spacing_nm": "later"}, "numeric or 'auto'"),
        ({"d_spacing_nm": -1.0}, "finite and positive"),
        ({"d_spacing_points": 1}, "d_spacing_points"),
    ],
)
def test_config_post_init_validation(tmp_path: Path, overrides: dict, match: str) -> None:
    """Cross-field validation rejects the obvious mistakes."""

    with pytest.raises(ValueError, match=match):
        _config(tmp_path, **overrides)


# --------------------------------------------------------------------------- #
# _build_grating                                                               #
# --------------------------------------------------------------------------- #
def test_build_grating_wires_multilayer_stack(tmp_path: Path) -> None:
    """The grating carries a multilayer stack with the requested geometry."""

    config = _config(tmp_path, n_bilayers=30)
    grating = mo._build_grating(config, d_spacing_nm=2.7, gamma=0.45, blaze_angle_deg=1.2)
    assert isinstance(grating, grax.BlazedGrating)
    assert grating.blaze_angle_deg == pytest.approx(1.2)
    assert grating.anti_blaze_angle_deg is None  # anti_blaze_angle_deg == 0.0 -> omitted
    stack = grating.coating_stack
    assert isinstance(stack, grax.MultilayerStack)
    assert stack.d_period_nm == pytest.approx(2.7)
    assert stack.gamma == pytest.approx(0.45)
    assert stack.n_bilayers == 30
    assert stack.top_material.name == "C"


def test_build_grating_passes_anti_blaze_when_set(tmp_path: Path) -> None:
    """A non-zero anti-blaze angle reaches the grating."""

    grating = mo._build_grating(
        _config(tmp_path, anti_blaze_angle_deg=3.0),
        d_spacing_nm=2.7,
        gamma=0.5,
        blaze_angle_deg=1.0,
    )
    assert grating.anti_blaze_angle_deg == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# Stage 0                                                                      #
# --------------------------------------------------------------------------- #
def test_run_d_spacing_study_writes_geometry_suggestion(fakes: None, tmp_path: Path) -> None:
    """Stage 0 stores the geometry d as the suggestion and the argmax separately."""

    config = _config(tmp_path)
    result = run_d_spacing_study(config)

    direct_angle = float(
        grax.monochromator_grazing_angles_deg(
            [9000.0], period_lpermm=2400.0, diffraction_order=2, cff=2.25
        )[0]
    )
    assert result.geometry_grazing_angle_deg == pytest.approx(direct_angle)
    assert result.d_suggested_nm == pytest.approx(round(result.geometry_d_nm, 1))

    state = json.loads(config.state_path.read_text(encoding="utf-8"))
    assert state["d_suggested_nm"] == pytest.approx(round(result.geometry_d_nm, 1))
    assert "d_reflectivity_best_nm" in state

    target_rows = result.results[np.isclose(result.results["energy_ev"], 9000.0)]
    best_row = target_rows.loc[target_rows["peak_rp"].idxmax()]
    assert result.d_reflectivity_best_nm == pytest.approx(best_row["d_spacing_nm"])

    assert result.combined_csv_path.is_file()
    assert result.plot_path.is_file()


# --------------------------------------------------------------------------- #
# Stage 1                                                                      #
# --------------------------------------------------------------------------- #
def test_run_gamma_study_selects_peak_and_leaves_config(fakes: None, tmp_path: Path) -> None:
    """Stage 1 picks the argmax gamma and does not touch the config or numeric d."""

    config = _config(tmp_path, d_spacing_nm=2.7)
    result = run_gamma_study(config)

    assert result.d_spacing_nm == pytest.approx(2.7)
    assert result.gamma_suggested == pytest.approx(0.5)
    state = json.loads(config.state_path.read_text(encoding="utf-8"))
    assert state["gamma_suggested"] == pytest.approx(0.5)
    # The config object is frozen and unchanged.
    assert config.gamma == 0.5
    assert config.d_spacing_nm == 2.7


def test_run_gamma_study_auto_reads_state(fakes: None, tmp_path: Path) -> None:
    """``d_spacing_nm="auto"`` resolves the stage-0 suggestion from the state file."""

    config = _config(tmp_path)  # d_spacing_nm defaults to "auto"
    run_d_spacing_study(config)
    suggested = json.loads(config.state_path.read_text(encoding="utf-8"))["d_suggested_nm"]
    result = run_gamma_study(config)
    assert result.d_spacing_nm == pytest.approx(suggested)


# --------------------------------------------------------------------------- #
# Stage 2                                                                      #
# --------------------------------------------------------------------------- #
def test_run_blaze_study_selects_best_blaze(fakes: None, tmp_path: Path) -> None:
    """Stage 2 runs one sweep per blaze value and picks the target-energy argmax."""

    config = _config(tmp_path, d_spacing_nm=2.7, gamma=0.5)
    result = run_blaze_study(config)

    assert result.blaze_suggested_deg == pytest.approx(1.1)
    assert result.gamma == pytest.approx(0.5)
    state = json.loads(config.state_path.read_text(encoding="utf-8"))
    assert state["blaze_suggested_deg"] == pytest.approx(1.1)
    assert sorted(set(result.results["blaze_angle_deg"])) == pytest.approx([0.9, 1.1, 1.3])
    assert result.combined_csv_path.is_file()
    assert result.plot_path.is_file()


def test_full_pipeline_state_accretes(fakes: None, tmp_path: Path) -> None:
    """Running all three stages leaves one state file with every suggestion."""

    config = _config(tmp_path)
    run_d_spacing_study(config)
    run_gamma_study(config)
    run_blaze_study(config)
    state = json.loads(config.state_path.read_text(encoding="utf-8"))
    assert {"d_suggested_nm", "gamma_suggested", "blaze_suggested_deg"} <= set(state)
