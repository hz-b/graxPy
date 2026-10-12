"""Pulse-picker geometry, coupled models, and saved web workflows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from grax.pulse_picker import (
    PulsePickerGrating,
    pulse_picker_rcwa_amplitudes,
    simulate_pulse_picker_rocking_curve,
    simulate_pulse_picker_timing,
)
from grax.web.persistence import GratingStore, build_grating_from_spec, grating_to_spec


def test_paper_geometry_and_components() -> None:
    """Check paper periods and independent switch-off of both gratings."""
    grating = PulsePickerGrating.paper_preset("LNB").at_voltage(9)
    assert grating.period_nm == pytest.approx(4000)
    assert grating.electrode_mask(np.array([500, 1500, 2500, 3500])).tolist() == [
        True,
        False,
        True,
        False,
    ]
    assert np.ptp(grating.displacement_nm(np.linspace(0, 4000, 1001))) == pytest.approx(
        grating.displacement_pv_nm
    )
    assert not grating.with_component("deformation").aluminium_on()
    assert not grating.with_component("aluminium").displacement_on()
    assert grating.at_voltage(0).profile_depth_nm() == 0
    lgs = PulsePickerGrating.paper_preset("LGS")
    assert lgs.period_nm == pytest.approx(2800)
    assert lgs.electrode_pitch_nm == 1400
    assert lgs.displacement_per_volt_nm == 0
    with pytest.raises(ValueError, match="complex Bragg reference"):
        simulate_pulse_picker_rocking_curve(lgs)


def test_persistence_round_trip() -> None:
    """Preserve the calibrated picker parameters in a saved specification."""
    original = PulsePickerGrating.paper_preset("LNB").at_voltage(9)
    spec = grating_to_spec(original, name="Picker")
    loaded = build_grating_from_spec(spec)
    assert isinstance(loaded, PulsePickerGrating)
    assert loaded.sample == "LNB"
    assert loaded.displacement_pv_nm == pytest.approx(original.displacement_pv_nm)


def test_rcwa_passivity_and_convergence() -> None:
    """Check passivity and basic Fourier-order convergence of all geometries."""
    grating = PulsePickerGrating.paper_preset("LNB").at_voltage(9)
    totals = {}
    for component in ("deformation", "aluminium", "combined"):
        candidate = grating.with_component(component)
        order, _, _, reflected, transmitted = pulse_picker_rcwa_amplitudes(
            candidate, fourier_orders=5
        )
        assert len(order) == 11
        assert np.all(reflected >= -1e-12)
        assert np.all(transmitted >= -1e-12)
        totals[component] = float(np.sum(reflected + transmitted))
        assert totals[component] <= 1.00001
    assert totals["combined"] < totals["deformation"]
    _, _, _, r3, t3 = pulse_picker_rcwa_amplitudes(grating, fourier_orders=3)
    assert abs(float(np.sum(r3 + t3)) - totals["combined"]) < 5e-4


def test_rocking_satellites_and_timing() -> None:
    """Check satellite parity, period, and effective bunch selection."""
    grating = PulsePickerGrating.paper_preset("LNB")
    grid = np.linspace(-0.012, 0.012, 241)
    off = simulate_pulse_picker_rocking_curve(
        grating, angle_offset_deg=grid, max_order=2, fourier_orders=5
    )
    on = simulate_pulse_picker_rocking_curve(
        grating.at_voltage(9), angle_offset_deg=grid, max_order=2, fourier_orders=5
    )
    assert off.satellite_spacing_deg == pytest.approx(0.003919, abs=2e-5)
    assert off.intensity[off.orders == 1].max() < 1e-7
    assert on.intensity[on.orders == 1].max() > 1e-3
    assert off.intensity[off.orders == 2].max() > 0
    timing = simulate_pulse_picker_timing(
        np.array([0, 9]),
        np.array([0, 0.15]),
        neighbour_count=1,
    )
    assert timing.bunch_times_ns.tolist() == [-2, 0, 2]
    assert timing.selected_intensity == pytest.approx(0.15)
    assert timing.neighbour_leakage < 1e-5


def test_web_create_edit_and_both_runs(tmp_path: Path) -> None:
    """Create, edit, and launch both workflows using HTTP forms."""
    from grax.web.app import create_app

    client = create_app(data_dir=tmp_path).test_client()
    fields = {
        "name": "Paper LNB",
        "grating_type": "pulse_picker",
        "pulse_sample": "LNB",
        "grating_period_nm": "4000",
        "electrode_pitch_nm": "2000",
        "finger_width_nm": "1000",
        "aluminium_thickness_nm": "100",
        "electrode_offset_nm": "0",
        "displacement_pv_nm": "0",
        "displacement_per_volt_nm": "0.0104",
        "displacement_harmonics": "",
        "x_resolution_nm": "40",
        "z_resolution_nm": "0.05",
    }
    assert client.post("/gratings", data=fields).status_code == 302
    store = GratingStore(tmp_path / "saved_gratings")
    saved = store.list()[0]
    grating_id = saved["id"]
    assert b"pulse_picker_rcwa" in client.get(f"/gratings/{grating_id}").data
    edit = client.get(f"/gratings/{grating_id}/edit")
    assert edit.status_code == 200
    assert b"4000" in edit.data
    fields["name"] = "Edited paper LNB"
    assert client.post(f"/gratings/{grating_id}", data=fields).status_code == 302
    assert store.load(grating_id)["name"] == "Edited paper LNB"

    common = {
        "energy_ev": "8000",
        "voltage_v": "9",
        "displacement_per_volt_nm": "0.0104",
        "fourier_orders": "3",
        "max_order": "2",
        "polarization": "s",
    }
    for workflow, plot in (
        ("pulse_picker_rocking", "rocking_curve.png"),
        ("pulse_picker_rcwa", "rcwa_orders.png"),
    ):
        response = client.post(
            f"/gratings/{grating_id}/runs", data={**common, "workflow": workflow}
        )
        assert response.status_code == 302
        run_id = response.headers["Location"].rstrip("/").split("/")[-1]
        run_dir = tmp_path / "runs" / run_id
        assert (run_dir / plot).is_file()
        assert json.loads((run_dir / "manifest.json").read_text())["status"] == "completed"
        assert client.get(f"/runs/{run_id}").status_code == 200
