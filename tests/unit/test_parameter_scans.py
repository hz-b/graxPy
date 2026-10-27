"""Energy-scan grouping, mounting geometry, and parameter-slice exports."""

import csv
import io
import json
import threading
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest
from werkzeug.datastructures import MultiDict

from grax import simulation
from grax.web import app as web_app
from grax.web import parameter_scans as scans
from grax.web import run_jobs, run_plots
from grax.web.routes import runs as run_routes
from grax.web.custom_grid import default_configuration
from grax.web.persistence import build_grating_from_spec


def spec():
    """Create a small grating with inexpensive numerical resolution."""
    return {
        "name": "Laminar",
        "grating_type": "laminar",
        "period_lpermm": 400,
        "x_resolution_nm": 1.0,
        "z_resolution_nm": 0.5,
        "width_to_period_ratio": 0.5,
        "depth_nm": 15,
        "left_wall_angle_deg": 15,
        "right_wall_angle_deg": 15,
        "roughness": None,
        "stack": {"type": "bare", "substrate_material": "Si"},
    }


def config():
    """Vary duty cycle and material as independent parameter axes."""
    value = default_configuration(spec())
    value["independent"] = [
        {"parameter": "width_to_period_ratio", "mode": "list", "values": [0.5, 0.6]},
        {"parameter": "stack.substrate_material", "mode": "list", "values": ["Si", "Au"]},
    ]
    return value


def build(value=None, mode="energy_sweep"):
    """Generate a small grouped scan for execution and export checks."""
    return scans.build_scans(
        value or config(),
        spec(),
        mode=mode,
        energies=np.array([100.0, 110.0, 120.0]),
        defaults={"grazing_angle_deg": 1.5, "fourier_orders": 5, "cff": 2.25,
                  "total_deviation_deg": 174.0},
    )


@pytest.mark.parametrize("mode", [
    "energy_sweep", "angle_sweep", "fixed_angle", "monochromator", "constant_deviation",
])
def test_none_coating_scan_is_supported_in_every_parameter_scan_mode(mode):
    """All grouped scan modes consume the shared None-material representation."""
    base = spec()
    base["stack"] = {
        "type": "single_layer", "substrate_material": "Si", "layer_material": "Au",
        "layer_thickness_nm": 20,
    }
    value = default_configuration(base)
    value["independent"] = [{
        "parameter": "stack.layer_material", "mode": "list", "values": ["None", "Ni"],
    }]

    sweep, cases = scans.build_scans(
        value, base, mode=mode, energies=np.array([100.0]),
        defaults={"grazing_angle_deg": 1.5, "fourier_orders": 5, "cff": 2.25,
                  "total_deviation_deg": 174.0},
    )

    assert len(sweep["scans"]) == 2
    assert cases[0]["scan_grating_spec"]["stack"]["type"] == "bare"
    assert cases[1]["scan_grating_spec"]["stack"]["layer_material"]["name"] == "Ni"


def fake_results(cases):
    """Assign distinct efficiencies to identify each exported scan column."""
    return [
        SimpleNamespace(
            case_id=case["case_id"],
            index=index,
            case_data=case,
            energy_ev=case["energy_ev"],
            grazing_angle_deg=case["grazing_angle_deg"],
            status="ok",
            error_message=None,
            selected_efficiency=0.1 * (index + 1),
            orders=np.array([-1, 0, 1]),
            efficiency_all=np.array([0.02, 0.01, 0.1 * (index + 1)]),
            diffraction_angle_all=np.array([-2.0, 1.0, 2.0]),
        )
        for index, case in enumerate(cases)
    ]


def test_cartesian_combinations_each_get_a_full_stable_scan():
    """Keep combinations separate with stable point identities."""
    sweep, cases = build()
    assert len(sweep["scans"]) == 4
    assert len(cases) == 12

    assert len({case["case_id"] for case in cases}) == 12
    assert [case["energy_ev"] for case in cases[:3]] == [100.0, 110.0, 120.0]
    assert all(case["grazing_angle_deg"] == 1.5 for case in cases)
    assert [case["case_id"] for case in build()[1]] == [case["case_id"] for case in cases]
    assert sweep["scans"][2]["grating_spec"]["width_to_period_ratio"] == 0.6


def test_coupled_rows_act_as_one_axis():
    """Pair dependent values without creating an extra Cartesian axis."""
    value = config()
    value["independent"] = value["independent"][1:]
    value["coupled_groups"] = [
        {
            "name": "Profile",
            "parameters": ["width_to_period_ratio", "depth_nm"],
            "rows": [
                {"width_to_period_ratio": 0.5, "depth_nm": 10},
                {"width_to_period_ratio": 0.6, "depth_nm": 20},
            ],
        }
    ]
    sweep, cases = build(value)
    assert len(cases) == 12
    assert sweep["axes"][1]["parameters"] == ["width_to_period_ratio", "depth_nm"]
    assert {
        (scan["parameters"]["width_to_period_ratio"], scan["parameters"]["depth_nm"])
        for scan in sweep["scans"]
    } == {(0.5, 10), (0.6, 20)}


def test_monochromator_recomputes_geometry_per_energy_and_line_density():
    """Match ordinary mounting calculations for every resolved grating."""
    value = default_configuration(spec())
    value["independent"] = [
        {"parameter": "period_lpermm", "mode": "list", "values": [400, 600]},
        {"parameter": "cff", "mode": "list", "values": [2.0, 2.25]},
    ]
    sweep, cases = build(value, "monochromator")
    for index, scan in enumerate(sweep["scans"]):
        grating = cases[index * 3]["grating"]
        reference = list(
            simulation.monochromator_cases(
                grating=grating,
                energies_ev=[100, 110, 120],
                cff=scan["settings"]["cff"],
                diffraction_order=1,
            )
        )
        assert [case["grazing_angle_deg"] for case in cases[index * 3 : index * 3 + 3]] == [
            case["grazing_angle_deg"] for case in reference
        ]
    assert cases[0]["grazing_angle_deg"] != cases[6]["grazing_angle_deg"]


def test_constant_deviation_keeps_beam_deviation_fixed_and_solves_grating_equation():
    """Keep 2θ and the grating equation satisfied at each sampled energy."""
    cases = list(simulation.constant_deviation_cases(
        grating=build_grating_from_spec(spec()),
        energies_ev=[200.0, 400.0, 800.0],
        total_deviation_deg=174.0,
    ))
    assert len(cases) == 3
    for case in cases:
        alpha = np.deg2rad(case["grazing_angle_deg"])
        beta = np.deg2rad(case["diffraction_angle_deg"])
        assert np.rad2deg(alpha + beta) == pytest.approx(6.0)
        wavelength_nm = 1239.8 / case["energy_ev"]
        period_nm = 1e6 / spec()["period_lpermm"]
        assert period_nm * (np.cos(alpha) - np.cos(beta)) == pytest.approx(wavelength_nm)


def test_constant_deviation_rejects_unphysical_angles():
    """Reject energy and deviation combinations without physical grazing angles."""
    grating = build_grating_from_spec(spec())
    with pytest.raises(ValueError, match="No (real|physical) grazing angles"):
        list(simulation.constant_deviation_cases(
            grating=grating, energies_ev=[10.0], total_deviation_deg=174.0,
        ))


def test_constant_deviation_parameter_scan_reuses_solver_and_resolved_grating():
    """Keep parameter combinations and solver choices in the shared scan builder."""
    value = default_configuration(spec())
    value["independent"] = [
        {"parameter": "period_lpermm", "mode": "list", "values": [400, 600]},
    ]
    sweep, cases = scans.build_scans(
        value,
        spec(),
        mode="constant_deviation",
        energies=np.array([200.0, 300.0]),
        defaults={"total_deviation_deg": 170.0, "diffraction_order": 1,
                  "fourier_orders": 5, "grazing_angle_deg": 1.5,
                  "solver": "neviere", "polarization": "p"},
    )
    assert len(sweep["scans"]) == 2
    assert all(case["solver"] == "neviere" for case in cases)
    assert [case["energy_ev"] for case in cases[:2]] == [200.0, 300.0]


def test_constant_deviation_material_loop_includes_uncoated_grating():
    """A material sweep can include an uncoated case and still start."""
    coated = spec()
    coated["x_resolution_nm"] = 50.0
    coated["z_resolution_nm"] = 2.0
    coated["stack"] = {
        "type": "single_layer", "substrate_material": "Si",
        "layer_material": "Au", "layer_thickness_nm": 10.0,
        "layer_roughness_sigma_nm": 0.2,
    }
    value = default_configuration(coated)
    value["independent"] = [{
        "parameter": "stack.layer_material", "mode": "list",
        "values": [
            {"name": "None", "density_g_cm3": None, "roughness_sigma_nm": None},
            {"name": "Au", "density_g_cm3": None, "roughness_sigma_nm": 0.8},
        ],
    }]
    defaults = {
        "total_deviation_deg": 174.0, "diffraction_order": 1,
        "fourier_orders": 2, "grazing_angle_deg": 1.5,
        "solver": "rcwa", "polarization": "s",
    }
    sweep, cases = scans.build_scans(
        value, coated, mode="constant_deviation",
        energies=np.array([300.0]), defaults=defaults,
    )

    assert len(sweep["scans"]) == len(cases) == 2
    assert [case["scan_grating_spec"]["stack"]["type"] for case in cases] == ["bare", "single_layer"]
    assert cases[0]["scan_grating_spec"]["stack"]["substrate_material"] == "Si"
    assert cases[1]["scan_grating_spec"]["stack"]["layer_roughness_sigma_nm"] == 0.8
    results = [simulation.run_simulation(
        grating=case["grating"], energy_ev=case["energy_ev"],
        grazing_angle_deg=case["grazing_angle_deg"],
        diffraction_order=case["diffraction_order"],
        fourier_orders=case["fourier_orders"],
        polarization=case["polarization"], solver=case["solver"],
        validate_physical_results=False,
    ) for case in cases]
    assert all(np.isfinite(result.selected_efficiency) for result in results)


def test_forbidden_mount_parameters_are_rejected():
    """Keep energy and incompatible mount settings out of variation axes."""
    value = config()
    value["independent"] = [{"parameter": "energy_ev", "mode": "list", "values": [100, 200]}]
    with pytest.raises(ValueError, match="not available"):
        build(value)


def test_exports_slice_other_axes_and_keep_missing_energy_rows(tmp_path):
    """Retain energy ordering and missing values in each parameter slice."""
    sweep, cases = build()
    results = fake_results(cases)
    results.pop(1)
    scans.write_outputs(list(reversed(results)), sweep, tmp_path, title="Sweep", include_plot=False)
    archive = scans.export_tables(
        tmp_path,
        sweep,
        orders=[1],
        axis_ids=[axis["id"] for axis in sweep["axes"]],
        formats={"dat", "csv"},
    )
    with zipfile.ZipFile(io.BytesIO(archive)) as zip_file:
        names = [name for name in zip_file.namelist() if name.endswith(".dat")]
        assert len(names) == 4
        table = list(csv.reader(io.StringIO(zip_file.read(names[0]).decode()), delimiter="\t"))
        assert len(table) == 4
        assert len(table[0]) == 3
        assert [row[0] for row in table[1:]] == ["100.0", "110.0", "120.0"]
        assert table[2][1] == "NaN"
        assert len([name for name in zip_file.namelist() if name.endswith(".yaml")]) == 4


def test_grouped_preview_keeps_scan_identities_and_selected_styles(tmp_path):
    """Keep scan-specific curves and styles distinct in comparisons."""
    sweep, cases = build()
    run_dir = tmp_path / "runs" / "sweep"
    run_dir.mkdir(parents=True)
    manifest = {
        "id": "sweep",
        "workflow": "fixed_angle",
        "display_name": "Sweep",
        "parameter_sweep": sweep,
        "run_input": {"diffraction_order": 1},
        "status": "completed",
        "artifacts": [],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    scans.write_outputs(fake_results(cases), sweep, run_dir, title="Sweep", include_plot=False)
    assert web_app._available_orders(run_dir) == [-1, 0, 1]
    client = web_app.create_app(data_dir=tmp_path).test_client()
    form = MultiDict(
        [
            ("run_ids", "sweep"),
            ("orders_sweep", "1"),
            ("scan_selection_sweep", "1"),
            ("scans_sweep", "scan-000000"),
            ("scans_sweep", "scan-000001"),
            ("line_style_sweep__scan-000001__order_1", "dash"),
        ]
    )
    payload = client.post("/_preview/plot", data=form).get_json()
    assert payload["ok"]
    figure = json.loads(payload["figure_json"])
    assert len(figure["data"]) == 2
    assert figure["data"][1]["line"]["dash"] == "dash"
    assert len({item["series_key"] for item in payload["series_controls"]}) == 2
    detail = client.get("/runs/sweep")
    assert detail.status_code == 200
    detail_html = detail.get_data(as_text=True)
    assert "Compare parameter energy scans" not in detail_html
    assert 'data-run-plot-image' in detail_html
    assert detail_html.index("data-live-run-monitor") < detail_html.index(
        'class="scan-list scan-list-scroll"'
    )
    exported = client.post(
        "/runs/sweep/export/energy-tables",
        data={"orders": ["1"], "axes": ["width_to_period_ratio"], "formats": ["dat"]},
    )
    assert exported.status_code == 200
    assert zipfile.is_zipfile(io.BytesIO(exported.data))


def test_queue_persists_normalized_combinations_and_counts_all_points(tmp_path, monkeypatch):
    """Preserve the scan snapshot and execution options through queue and resume."""
    launched = {}
    monkeypatch.setattr(run_jobs, "_start_run_worker", lambda **kwargs: launched.update(kwargs))
    # The resume route starts workers itself, so patch its lookup too.
    monkeypatch.setattr(run_routes, "_start_run_worker", lambda **kwargs: launched.update(kwargs))
    app = web_app.create_app(data_dir=tmp_path)
    form = MultiDict(
        {
            "workflow": "angle_sweep",
            "energy_ev": "100",
            "grazing_angle_start_deg": "1",
            "grazing_angle_stop_deg": "2",
            "grazing_angle_points": "3",
            "diffraction_order": "1",
            "fourier_orders": "5",
            "parameter_sweep_config": json.dumps(config()),
            "max_workers_mode": "manual",
            "max_workers": "1",
        }
    )
    from grax.web.persistence import build_grating_from_spec

    manifest = web_app._queue_run(
        app=app,
        data_dir=tmp_path,
        grating_id="g",
        grating_name="Test",
        grating_spec=spec(),
        grating=build_grating_from_spec(spec()),
        form=form,
    )
    assert manifest["total_points"] == 12
    assert len(manifest["parameter_sweep"]["scans"]) == 4
    assert manifest["parameter_sweep"]["sample_axis"]["values"] == [1.0, 1.5, 2.0]
    assert web_app._parameter_scan_point_count(launched["form_data"], 1) == 12
    cases = web_app._cases_for_workflow(
        simulation=simulation,
        workflow="angle_sweep",
        grating=build_grating_from_spec(spec()),
        energies=launched["energies"],
        form=launched["form_data"],
        diffraction_order=1,
        fourier_orders=5,
    )
    assert len(cases) == 12
    assert {case["energy_ev"] for case in cases} == {100.0}
    assert {case["grazing_angle_deg"] for case in cases} == {1.0, 1.5, 2.0}
    assert [case["scan_id"] for case in cases[:3]] == ["scan-000000"] * 3
    assert [case["case_id"] for case in cases[:3]] == [
        f"scan-000000-angle-{index:06d}" for index in range(3)
    ]
    path = tmp_path / "runs" / manifest["id"] / "manifest.json"
    saved = json.loads(path.read_text())
    saved["status"] = "aborted"
    path.write_text(json.dumps(saved))
    response = app.test_client().post(f"/runs/{manifest['id']}/resume")
    assert response.status_code == 302
    assert launched["resume"] is True
    assert launched["form_data"]["parameter_sweep_config"] == manifest["parameter_sweep"]["grid"]


def test_monochromator_run_keeps_multidict_form_for_run_input(tmp_path, monkeypatch):
    """Monochromator CFF preprocessing must not break form-list access."""
    launched = {}
    monkeypatch.setattr(run_jobs, "_start_run_worker", lambda **kwargs: launched.update(kwargs))
    app = web_app.create_app(data_dir=tmp_path)
    form = MultiDict(
        [
            ("workflow", "monochromator"),
            ("energy_start_ev", "100"),
            ("energy_stop_ev", "120"),
            ("energy_points", "3"),
            ("cff", "2.25"),
            ("diffraction_order", "1"),
            ("fourier_orders", "5"),
            ("polarization", "s"),
            ("orders", "-1"),
            ("orders", "1"),
            ("parameter_sweep_config", json.dumps(default_configuration(spec()))),
        ]
    )
    from grax.web.persistence import build_grating_from_spec

    manifest = web_app._queue_run(
        app=app,
        data_dir=tmp_path,
        grating_id="mono-grating",
        grating_name="Monochromator test",
        grating_spec=spec(),
        grating=build_grating_from_spec(spec()),
        form=form,
    )

    assert manifest["run_input"]["orders"] == ["-1", "1"]
    assert manifest["run_input"]["parameter_sweep_config"]
    assert launched["form_data"]["parameter_sweep_config"]


def test_angle_sweep_is_one_curve_with_ordered_angles_and_nan_gaps(tmp_path):
    form = {
        "workflow": "angle_sweep",
        "energy_ev": "100",
        "grazing_angle_start_deg": "1",
        "grazing_angle_stop_deg": "2",
        "grazing_angle_points": "11",
        "diffraction_order": "1",
        "fourier_orders": "5",
    }
    sweep, cases = web_app._parameter_scans_from_form(form, spec(), np.array([100.0]))
    assert len(sweep["scans"]) == 1
    assert len(cases) == 11
    assert [case["grazing_angle_deg"] for case in cases] == pytest.approx(np.linspace(1, 2, 11))
    assert len({case["scan_id"] for case in cases}) == 1
    results = fake_results(cases)
    results.pop(4)
    scans.write_outputs(list(reversed(results)), sweep, tmp_path, title="Angles", include_plot=False)
    values = scans.load_series(tmp_path, sweep, "scan-000000", 1)
    assert len(values) == 11
    assert np.isnan(values[4])
    archive = scans.export_tables(tmp_path, sweep, orders=[1], axis_ids=["scan"], formats={"csv", "dat"})
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        name = next(name for name in zipped.namelist() if name.endswith(".dat"))
        rows = list(csv.reader(io.StringIO(zipped.read(name).decode()), delimiter="\t"))
        assert rows[0][0] == "grazing_angle_deg"
        assert len(rows) == 12
        assert rows[5][1] == "NaN"


def test_legacy_angle_sweep_keeps_energy_sample_axis():
    value = default_configuration(spec())
    value["independent"] = [{"parameter": "grazing_angle_deg", "mode": "list", "values": [1, 2]}]
    sweep, cases = scans.build_scans(
        value, spec(), mode="angle_sweep", energies=np.array([100.0]),
        defaults={"grazing_angle_deg": 1.5, "fourier_orders": 5},
    )
    assert "sample_axis" not in sweep
    assert len(sweep["scans"]) == 2
    assert [case["scan_id"] for case in cases] == ["scan-000000", "scan-000001"]


def test_coupled_angle_variations_remain_one_curve_per_combination():
    value = default_configuration(spec())
    value["independent"] = [{"parameter": "stack.substrate_material", "mode": "list", "values": ["Si", "Au"]}]
    value["coupled_groups"] = [{
        "name": "Profile", "parameters": ["width_to_period_ratio", "depth_nm"],
        "rows": [
            {"width_to_period_ratio": 0.5, "depth_nm": 10},
            {"width_to_period_ratio": 0.6, "depth_nm": 20},
        ],
    }]
    form = {
        "workflow": "angle_sweep", "energy_ev": "100",
        "grazing_angle_start_deg": "1", "grazing_angle_stop_deg": "2", "grazing_angle_points": "3",
        "parameter_sweep_config": value,
    }
    sweep, cases = web_app._parameter_scans_from_form(form, spec(), np.array([100.0]))
    assert len(sweep["axes"]) == 2
    assert len(sweep["scans"]) == 4
    assert len(cases) == 12
    assert all([case["grazing_angle_deg"] for case in cases[index:index + 3]] == [1.0, 1.5, 2.0] for index in range(0, 12, 3))


def test_angle_comparison_uses_grazing_angle_axis(tmp_path):
    form = {
        "workflow": "angle_sweep", "energy_ev": "100",
        "grazing_angle_start_deg": "1", "grazing_angle_stop_deg": "2", "grazing_angle_points": "3",
    }
    sweep, cases = web_app._parameter_scans_from_form(form, spec(), np.array([100.0]))
    run_dir = tmp_path / "runs" / "angles"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({
        "id": "angles", "display_name": "Angles", "workflow": "angle_sweep",
        "parameter_sweep": sweep,
    }))
    scans.write_outputs(fake_results(cases), sweep, run_dir, title="Angles", include_plot=False)
    _, series = web_app._load_selected_plot_series(
        data_dir=tmp_path, run_ids=["angles"], order_selection={"angles": [1]},
    )
    assert len(series) == 1
    assert series[0]["energies"] == [1.0, 1.5, 2.0]
    assert series[0]["x_axis_id"] == "grazing_angle_deg"
    _, figure_json, _ = web_app._build_plotly_compare_payload(
        data_dir=tmp_path, run_ids=["angles"], order_selection={"angles": [1]},
        plot_options={"title": "Angles", "x_axis_type": "linear", "y_axis_type": "linear"},
    )
    figure = json.loads(figure_json)
    assert figure["layout"]["xaxis"]["title"]["text"] == "Grazing angle (deg)"


def test_empty_scan_selection_defaults_to_all_scans_for_selected_run_and_order(tmp_path):
    sweep, cases = build()
    run_id = "empty-scan-selection"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({
        "id": run_id, "workflow": "energy_sweep", "status": "completed",
        "grating_name": "Laminar", "run_input": {"diffraction_order": 1},
        "parameter_sweep": sweep,
    }))
    scans.write_outputs(fake_results(cases), sweep, run_dir, title="Laminar", include_plot=False)

    _, series = web_app._load_selected_plot_series(
        data_dir=tmp_path,
        run_ids=[run_id],
        order_selection={run_id: [1]},
        scan_selection={run_id: []},
    )

    assert len(series) == len(sweep["scans"])


def test_grouped_live_plot_refreshes_and_saved_plot_repairs(tmp_path, monkeypatch):
    sweep, cases = build()
    run_id = "live-grouped"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({
        "id": run_id, "workflow": "energy_sweep", "status": "completed",
        "grating_name": "Laminar", "run_input": {"diffraction_order": 1},
        "parameter_sweep": sweep, "artifacts": [],
    }))
    results = fake_results(cases)
    state = web_app.ActiveRunState(
        run_id=run_id, workflow="energy_sweep", total_points=len(cases),
        worker_mode="manual", requested_workers=1, resolved_workers=1,
    )
    live_path = run_dir / "live_progress.png"
    assert web_app._publish_live_progress_snapshot(
        state=state, results=results[:1], output_path=live_path,
        diffraction_order=1, title="Live", now=1.0,
    )
    first_token = state.plot_token
    assert not web_app._publish_live_progress_snapshot(
        state=state, results=results[:2], output_path=live_path,
        diffraction_order=1, title="Live", now=1.5,
    )
    assert web_app._publish_live_progress_snapshot(
        state=state, results=results[:2], output_path=live_path,
        diffraction_order=1, title="Live", now=3.0,
    )
    assert state.plot_token != first_token
    assert live_path.is_file()
    failed = SimpleNamespace(**vars(results[0]))
    failed.status = "failed"
    failed_state = web_app.ActiveRunState(
        run_id=run_id, workflow="energy_sweep", total_points=len(cases),
        worker_mode="manual", requested_workers=1, resolved_workers=1,
    )
    assert web_app._publish_live_progress_snapshot(
        state=failed_state, results=[failed], output_path=run_dir / "failed_progress.png",
        diffraction_order=1, title="Failed first point", now=1.0,
    )

    monkeypatch.setattr(run_plots, "_load_checkpoint_results", lambda path: results)
    client = web_app.create_app(data_dir=tmp_path).test_client()
    page = client.get(f"/runs/{run_id}")
    final_path = run_dir / "selected_efficiency.png"
    assert page.status_code == 200
    assert b"data-run-plot-image" in page.data
    assert final_path.is_file()
    first_mtime = final_path.stat().st_mtime_ns
    client.get(f"/runs/{run_id}")
    assert final_path.stat().st_mtime_ns == first_mtime
    scans.write_outputs(results, sweep, run_dir, title="Saved", include_plot=False)
    final_path.unlink()
    monkeypatch.setattr(run_plots, "_load_checkpoint_results", lambda path: [])
    assert client.get(f"/runs/{run_id}").status_code == 200
    assert final_path.is_file()


def test_energy_sweep_keeps_angle_fixed_across_parameter_combinations(tmp_path, monkeypatch):
    """Combine optional parameters with the energy range at one grazing angle."""
    launched = {}
    monkeypatch.setattr(run_jobs, "_start_run_worker", lambda **kwargs: launched.update(kwargs))
    app = web_app.create_app(data_dir=tmp_path)
    form = MultiDict(
        {
            "workflow": "energy_sweep",
            "energy_start_ev": "100",
            "energy_stop_ev": "120",
            "energy_points": "3",
            "grazing_angle_deg": "1.5",
            "diffraction_order": "1",
            "fourier_orders": "5",
            "parameter_sweep_config": json.dumps(config()),
            "max_workers_mode": "manual",
            "max_workers": "1",
        }
    )
    from grax.web.persistence import build_grating_from_spec

    manifest = web_app._queue_run(
        app=app,
        data_dir=tmp_path,
        grating_id="energy-grating",
        grating_name="Energy test",
        grating_spec=spec(),
        grating=build_grating_from_spec(spec()),
        form=form,
    )
    assert manifest["parameter_sweep"]["mode"] == "energy_sweep"
    assert manifest["total_points"] == 12
    assert launched["energies"].tolist() == [100.0, 110.0, 120.0]
    cases = web_app._cases_for_workflow(
        simulation=simulation,
        workflow="energy_sweep",
        grating=build_grating_from_spec(spec()),
        energies=launched["energies"],
        form=launched["form_data"],
        diffraction_order=1,
        fourier_orders=5,
    )
    assert len(cases) == 12
    assert {case["grazing_angle_deg"] for case in cases} == {1.5}
    assert {case["energy_ev"] for case in cases} == {100.0, 110.0, 120.0}


def test_checkpoint_resume_finishes_remaining_points_and_rejects_changes(tmp_path):
    """Resume only pending points and reject changed calculation inputs."""
    _, cases = build()
    stop = threading.Event()
    runner = simulation.BatchSimulationRunner(
        max_workers=1, checkpoint_dir=tmp_path, stop_event=stop, show_progress=False
    )
    iterator = runner.run_cases(cases)
    first = next(iterator)
    assert first.status == "ok"
    stop.set()
    list(iterator)
    resumed = simulation.BatchSimulationRunner(
        max_workers=1, checkpoint_dir=tmp_path, resume=True, show_progress=False
    )
    remaining = list(resumed.run_cases(build()[1]))
    assert len(remaining) == len(cases) - 1
    assert first.case_id not in {result.case_id for result in remaining}
    changed = build()[1]
    changed[0]["energy_ev"] = 101.0
    with pytest.raises(ValueError, match="(?i)(checkpoint|resume)"):
        list(resumed.run_cases(changed))
