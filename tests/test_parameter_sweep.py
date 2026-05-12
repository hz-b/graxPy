from __future__ import annotations

from pathlib import Path

import numpy as np

from grax import (
    BlazedGrating,
    get_default_parameter_study_ranges,
    plot_parameter_study,
    run_parameter_study,
)
from grax.simulation import RCWASimulation


def test_run_parameter_study_returns_expected_shapes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_run_single(self, photon_energy_ev: float) -> dict[str, object]:
        if self.grating.x_resolution_nm <= 0.11 and photon_energy_ev == 600.0:
            raise ValueError("synthetic failure")
        efficiency = (
            0.001 * photon_energy_ev
            + 0.01 * self.fourier_orders
            + 0.1 * float(self.grating.x_resolution_nm)
            + 0.2 * float(self.grating.z_resolution_nm)
        )
        return {
            "orders": np.asarray([-1, 0, 1], dtype=int),
            "efficiency": float(efficiency),
            "diffraction_angle_deg": 1.0,
            "efficiency_all": np.asarray([efficiency, 0.5, 0.0], dtype=float),
            "diffraction_angle_all": np.asarray([1.0, 2.0, 3.0], dtype=float),
        }

    monkeypatch.setattr(RCWASimulation, "run_single", fake_run_single)

    grating = BlazedGrating(
        period_lpermm=600,
        blaze_angle_deg=0.75,
        x_resolution_nm=0.5,
        z_resolution_nm=0.1,
    )
    fourier_values = np.arange(5, 10, 2, dtype=int)
    x_values = np.geomspace(10.0, 0.1, 4)
    z_values = np.geomspace(10.0, 0.1, 4)

    result = run_parameter_study(
        grating=grating,
        energies_ev=[100.0, 600.0],
        grazing_angle_deg=1.5,
        fourier_orders_values=fourier_values,
        x_resolution_values=x_values,
        z_resolution_values=z_values,
        output_dir=tmp_path,
        save_csv=True,
        show_progress=False,
    )

    assert np.array_equal(result.energies_ev, np.asarray([100.0, 600.0], dtype=float))
    assert result.grazing_angle_deg == 1.5
    assert len(result.results) == 2
    assert np.array_equal(result.fourier_orders_values, fourier_values)
    assert np.allclose(result.x_resolution_values, x_values)
    assert np.allclose(result.z_resolution_values, z_values)

    first_energy = result.results[0]
    assert set(first_energy.sweeps) == {"fourier_orders", "x_resolution_nm", "z_resolution_nm"}
    assert first_energy.sweeps["fourier_orders"].efficiencies.shape == fourier_values.shape
    assert first_energy.sweeps["x_resolution_nm"].efficiencies.shape == x_values.shape
    assert first_energy.sweeps["z_resolution_nm"].efficiencies.shape == z_values.shape

    second_energy = result.results[1]
    assert second_energy.sweeps["x_resolution_nm"].errors[-1]

    expected_csv = tmp_path / "parameter_study_fourier_orders_E100.0eV.csv"
    assert expected_csv.exists()


def test_plot_parameter_study_writes_output(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_run_single(self, photon_energy_ev: float) -> dict[str, object]:
        efficiency = float(photon_energy_ev / 1000.0 + self.fourier_orders / 100.0)
        return {
            "orders": np.asarray([-1, 0, 1], dtype=int),
            "efficiency": efficiency,
            "diffraction_angle_deg": 1.0,
            "efficiency_all": np.asarray([efficiency, 0.5, 0.0], dtype=float),
            "diffraction_angle_all": np.asarray([1.0, 2.0, 3.0], dtype=float),
        }

    monkeypatch.setattr(RCWASimulation, "run_single", fake_run_single)

    grating = BlazedGrating(period_lpermm=600, blaze_angle_deg=0.75)
    result = run_parameter_study(
        grating=grating,
        energies_ev=[100.0],
        grazing_angle_deg=1.5,
        fourier_orders_values=[5, 7],
        x_resolution_values=[10.0, 1.0],
        z_resolution_values=[10.0, 1.0],
        save_csv=False,
        show_progress=False,
    )

    output_path = tmp_path / "parameter_study_grid.png"
    figure = plot_parameter_study(result, output_filename=output_path)

    assert figure is None
    assert output_path.exists()


def test_default_parameter_study_ranges_match_public_contract() -> None:
    fourier_values, x_values, z_values = get_default_parameter_study_ranges()

    assert np.array_equal(fourier_values, np.arange(5, 26, 2, dtype=int))
    assert len(x_values) == 10
    assert len(z_values) == 10
    assert x_values[0] == 10.0
    assert x_values[-1] == 0.1
    assert z_values[0] == 10.0
    assert z_values[-1] == 0.1


def test_run_parameter_study_supports_blazed_xz_scans(
    monkeypatch,
) -> None:
    def fake_run_single(self, photon_energy_ev: float) -> dict[str, object]:
        efficiency = float(self.grating.x_resolution_nm + self.grating.z_resolution_nm + photon_energy_ev / 1000.0)
        return {
            "orders": np.asarray([-1, 0, 1], dtype=int),
            "efficiency": efficiency,
            "diffraction_angle_deg": 1.0,
            "efficiency_all": np.asarray([efficiency, 0.5, 0.0], dtype=float),
            "diffraction_angle_all": np.asarray([1.0, 2.0, 3.0], dtype=float),
        }

    monkeypatch.setattr(RCWASimulation, "run_single", fake_run_single)

    grating = BlazedGrating(
        period_lpermm=600,
        blaze_angle_deg=0.75,
        x_resolution_nm=0.5,
        z_resolution_nm=0.1,
    )

    result = run_parameter_study(
        grating=grating,
        energies_ev=[100.0],
        grazing_angle_deg=1.5,
        fourier_orders_values=[5],
        x_resolution_values=[10.0, 1.0],
        z_resolution_values=[10.0, 1.0],
        save_csv=False,
        show_progress=False,
    )

    x_sweep = result.results[0].sweeps["x_resolution_nm"]
    z_sweep = result.results[0].sweeps["z_resolution_nm"]

    assert not np.any(x_sweep.errors)
    assert not np.any(z_sweep.errors)
    assert np.all(np.isfinite(x_sweep.efficiencies))
    assert np.all(np.isfinite(z_sweep.efficiencies))
