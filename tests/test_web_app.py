# ruff: noqa: D100,D103

from __future__ import annotations

import json
from pathlib import Path

import pytest
from werkzeug.datastructures import MultiDict

from grax.gratings import BlazedGrating, LaminarGrating
from grax.stacks import MultilayerStack
from grax.web.persistence import (
    GratingStore,
    build_grating_from_spec,
    grating_to_spec,
    load_material_catalog,
)


def _write_run_fixture(
    base_dir: Path,
    *,
    run_id: str,
    display_name: str,
    grating_name: str = "Demo grating",
    workflow: str = "fixed_angle",
    orders: tuple[int, ...] = (1, 2),
) -> None:
    run_dir = base_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "id": run_id,
        "created_at": "2026-06-10T12:00:00",
        "workflow": workflow,
        "grating_id": "grating-1",
        "grating_name": grating_name,
        "display_name": display_name,
        "status": "ok",
        "artifacts": ["all_orders.csv"],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with (run_dir / "all_orders.csv").open("w", encoding="utf-8", newline="") as handle:
        handle.write("case_id,energy_ev,grazing_angle_deg,order,efficiency,diffraction_angle_deg\n")
        for energy in (100.0, 110.0):
            for order in orders:
                handle.write(
                    f"case-{energy:.1f},{energy:.1f},1.5,{order},"
                    f"{0.05 * order + energy / 1000:.6f},{order * 1.2:.6f}\n"
                )


def test_saved_grating_round_trips_laminar_multilayer(tmp_path: Path) -> None:
    catalog = load_material_catalog()
    grating = LaminarGrating(
        period_lpermm=400,
        width_to_period_ratio=0.67,
        depth_nm=14.9,
        left_wall_angle_deg=15.0,
        right_wall_angle_deg=15.0,
        coating_stack=MultilayerStack(
            substrate_material=catalog["Si"],
            material_a=catalog["Cr"],
            material_b=catalog["C"],
            d_period_nm=6.5,
            gamma=0.45,
            n_bilayers=4,
            top_material=catalog["C"],
        ),
        x_resolution_nm=2.0,
        z_resolution_nm=0.5,
    )
    store = GratingStore(tmp_path / "gratings")

    saved = store.save(grating_to_spec(grating, name="Laminar ML"))
    loaded = build_grating_from_spec(store.load(saved["id"]), catalog)

    assert isinstance(loaded, LaminarGrating)
    assert loaded.period_lpermm == 400
    assert loaded.depth_nm == pytest.approx(14.9)
    assert isinstance(loaded.coating_stack, MultilayerStack)
    assert loaded.coating_stack.n_bilayers == 4
    assert loaded.coating_stack.d_period_nm == pytest.approx(6.5)


def test_saved_grating_round_trips_blazed_single_layer(tmp_path: Path) -> None:
    catalog = load_material_catalog()
    grating = BlazedGrating(
        period_lpermm=600,
        blaze_angle_deg=0.75,
        anti_blaze_angle_deg=5.597,
        substrate_material=catalog["Si"],
        layer_material=catalog["Au"],
        layer_thickness_nm=30.0,
        x_resolution_nm=1.5,
        z_resolution_nm=0.25,
    )
    store = GratingStore(tmp_path / "gratings")

    saved = store.save(grating_to_spec(grating, name="Blazed Au"))
    loaded = build_grating_from_spec(store.load(saved["id"]), catalog)

    assert isinstance(loaded, BlazedGrating)
    assert loaded.period_lpermm == 600
    assert loaded.blaze_angle_deg == pytest.approx(0.75)
    assert loaded.anti_blaze_angle_deg == pytest.approx(5.597)
    assert loaded.layer_thickness_nm == pytest.approx(30.0)


def test_grating_store_writes_plain_json(tmp_path: Path) -> None:
    store = GratingStore(tmp_path / "gratings")

    saved = store.save(
        {
            "name": "Demo",
            "grating_type": "laminar",
            "period_lpermm": 400,
            "x_resolution_nm": 1.0,
            "z_resolution_nm": 1.0,
            "width_to_period_ratio": 0.67,
            "depth_nm": 14.9,
            "left_wall_angle_deg": 15.0,
            "right_wall_angle_deg": 15.0,
            "stack": {
                "type": "single_layer",
                "substrate_material": "Si",
                "layer_material": "Pt",
                "layer_thickness_nm": 28.77,
            },
        }
    )

    payload = json.loads((tmp_path / "gratings" / f"{saved['id']}.json").read_text())

    assert payload["schema_version"] == 1
    assert payload["id"] == saved["id"]
    assert payload["name"] == "Demo"


def test_flask_app_creates_grating_and_lists_it(tmp_path: Path) -> None:
    flask = pytest.importorskip("flask")
    assert flask is not None

    from grax.web.app import create_app

    app = create_app(data_dir=tmp_path)
    client = app.test_client()

    response = client.post(
        "/gratings",
        data={
            "name": "Local laminar",
            "grating_type": "laminar",
            "period_lpermm": "400",
            "x_resolution_nm": "2.0",
            "z_resolution_nm": "0.5",
            "width_to_period_ratio": "0.67",
            "depth_nm": "14.9",
            "left_wall_angle_deg": "15.0",
            "right_wall_angle_deg": "15.0",
            "stack_type": "single_layer",
            "substrate_material": "Si",
            "layer_material": "Pt",
            "layer_thickness_nm": "28.77",
        },
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"Local laminar" in response.data
    assert len(GratingStore(tmp_path / "saved_gratings").list()) == 1


def test_index_mentions_result_locations(tmp_path: Path) -> None:
    pytest.importorskip("flask")

    from grax.web.app import create_app

    response = create_app(data_dir=tmp_path / ".grax-web").test_client().get("/")

    assert response.status_code == 200
    assert b".grax-web/runs/" in response.data
    assert b".grax-web/plots/" in response.data


def test_grating_form_exposes_conditional_profile_sections(tmp_path: Path) -> None:
    pytest.importorskip("flask")

    from grax.web.app import create_app

    app = create_app(data_dir=tmp_path)
    response = app.test_client().get("/gratings/new")

    assert response.status_code == 200
    assert b'data-grating-section="laminar"' in response.data
    assert b'data-grating-section="blazed"' in response.data
    assert b"web.js" in response.data


def test_flask_app_edits_saved_grating(tmp_path: Path) -> None:
    pytest.importorskip("flask")

    from grax.web.app import create_app

    app = create_app(data_dir=tmp_path)
    client = app.test_client()
    client.post(
        "/gratings",
        data={
            "name": "Before",
            "grating_type": "laminar",
            "period_lpermm": "400",
            "x_resolution_nm": "2.0",
            "z_resolution_nm": "0.5",
            "width_to_period_ratio": "0.67",
            "depth_nm": "14.9",
            "left_wall_angle_deg": "15.0",
            "right_wall_angle_deg": "15.0",
            "stack_type": "single_layer",
            "substrate_material": "Si",
            "layer_material": "Pt",
            "layer_thickness_nm": "28.77",
        },
    )
    grating_id = GratingStore(tmp_path / "saved_gratings").list()[0]["id"]

    response = client.post(
        f"/gratings/{grating_id}",
        data={
            "name": "After",
            "grating_type": "laminar",
            "period_lpermm": "450",
            "x_resolution_nm": "2.0",
            "z_resolution_nm": "0.5",
            "width_to_period_ratio": "0.6",
            "depth_nm": "12.0",
            "left_wall_angle_deg": "20.0",
            "right_wall_angle_deg": "20.0",
            "stack_type": "single_layer",
            "substrate_material": "Si",
            "layer_material": "Pt",
            "layer_thickness_nm": "28.77",
        },
        follow_redirects=True,
    )

    saved = GratingStore(tmp_path / "saved_gratings").load(grating_id)
    assert response.status_code == 200
    assert saved["name"] == "After"
    assert saved["period_lpermm"] == 450
    assert saved["depth_nm"] == pytest.approx(12.0)


def test_flask_app_runs_fixed_angle_sweep_with_saved_grating(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("flask")

    from grax.simulation.models import CaseExecutionResult
    from grax.web.app import create_app

    captured_cases = []

    def fake_run_cases(self, cases, metadata=None):  # type: ignore[no-untyped-def]
        for index, case in enumerate(cases):
            captured_cases.append(case)
            yield CaseExecutionResult(
                case_id=str(case["case_id"]),
                index=index,
                label=None,
                energy_ev=float(case["energy_ev"]),
                grazing_angle_deg=float(case["grazing_angle_deg"]),
                orders=__import__("numpy").asarray([-1, 0, 1]),
                selected_efficiency=0.25,
                selected_diffraction_angle_deg=1.2,
                efficiency_all=__import__("numpy").asarray([0.25, 0.1, 0.0]),
                diffraction_angle_all=__import__("numpy").asarray([1.2, 0.0, -1.2]),
                status="ok",
                case_data={key: value for key, value in case.items() if key != "grating"},
            )

    monkeypatch.setattr("grax.simulation.BatchSimulationRunner.run_cases", fake_run_cases)
    app = create_app(data_dir=tmp_path)
    client = app.test_client()
    create_response = client.post(
        "/gratings",
        data={
            "name": "Run grating",
            "grating_type": "blazed",
            "period_lpermm": "600",
            "x_resolution_nm": "2.0",
            "z_resolution_nm": "0.5",
            "blaze_angle_deg": "0.75",
            "anti_blaze_angle_deg": "",
            "stack_type": "single_layer",
            "substrate_material": "Si",
            "layer_material": "Au",
            "layer_thickness_nm": "30.0",
        },
    )
    assert create_response.status_code == 302
    grating_id = GratingStore(tmp_path / "saved_gratings").list()[0]["id"]

    run_response = client.post(
        f"/gratings/{grating_id}/runs",
        data={
            "workflow": "fixed_angle",
            "energy_start_ev": "100",
            "energy_stop_ev": "120",
            "energy_points": "3",
            "grazing_angle_deg": "1.5",
            "diffraction_order": "1",
            "fourier_orders": "5",
            "run_x_resolution_nm": "0.75",
            "run_z_resolution_nm": "0.25",
        },
        follow_redirects=True,
    )

    assert run_response.status_code == 200
    assert b"fixed_angle" in run_response.data
    assert (tmp_path / "runs").exists()
    assert captured_cases
    assert captured_cases[0]["x_resolution_nm"] == pytest.approx(0.75)
    run_id = next((path.parent.name for path in (tmp_path / "runs").glob("*/manifest.json")), None)
    assert run_id is not None
    manifest = json.loads((tmp_path / "runs" / run_id / "manifest.json").read_text())
    assert manifest["display_name"] == "Run grating · fixed_angle"


def test_plot_page_lists_saved_runs_and_orders(tmp_path: Path) -> None:
    pytest.importorskip("flask")

    from grax.web.app import create_app

    _write_run_fixture(tmp_path, run_id="run-1", display_name="Alpha run", orders=(1, 3))
    _write_run_fixture(tmp_path, run_id="run-2", display_name="Beta run", orders=(2,))

    app = create_app(data_dir=tmp_path)
    response = app.test_client().get("/plots")

    assert response.status_code == 200
    assert b"Alpha run" in response.data
    assert b"Order 3" in response.data
    assert b"Order 2" in response.data


def test_manage_runs_page_renames_and_deletes_selected_runs(tmp_path: Path) -> None:
    pytest.importorskip("flask")

    from grax.web.app import create_app

    _write_run_fixture(tmp_path, run_id="run-1", display_name="Alpha run")
    _write_run_fixture(tmp_path, run_id="run-2", display_name="Beta run")

    app = create_app(data_dir=tmp_path)
    client = app.test_client()

    page = client.get("/runs/manage")
    assert page.status_code == 200
    assert b"data-confirm=\"Delete the selected runs?" in page.data

    rename_response = client.post(
        "/runs/manage",
        data={
            "action": "save",
            "display_name_run-1": "Renamed run",
            "display_name_run-2": "Beta run",
        },
        follow_redirects=True,
    )
    assert rename_response.status_code == 200
    assert b"Renamed run" in rename_response.data

    delete_response = client.post(
        "/runs/manage",
        data={
            "action": "delete",
            "delete_run_id": ["run-2"],
        },
        follow_redirects=True,
    )
    assert delete_response.status_code == 200
    assert not (tmp_path / "runs" / "run-2").exists()


def test_flask_app_plots_selected_orders_across_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("flask")

    from grax.simulation.models import CaseExecutionResult
    from grax.web.app import create_app

    def fake_run_cases(self, cases, metadata=None):  # type: ignore[no-untyped-def]
        for index, case in enumerate(cases):
            yield CaseExecutionResult(
                case_id=str(case["case_id"]),
                index=index,
                label=None,
                energy_ev=float(case["energy_ev"]),
                grazing_angle_deg=float(case["grazing_angle_deg"]),
                orders=__import__("numpy").asarray([-2, -1, 1]),
                selected_efficiency=0.25,
                selected_diffraction_angle_deg=1.2,
                efficiency_all=__import__("numpy").asarray([0.15, 0.25, 0.05]),
                diffraction_angle_all=__import__("numpy").asarray([1.5, 1.2, 0.9]),
                status="ok",
                case_data={key: value for key, value in case.items() if key != "grating"},
            )

    monkeypatch.setattr("grax.simulation.BatchSimulationRunner.run_cases", fake_run_cases)
    app = create_app(data_dir=tmp_path)
    client = app.test_client()

    for name, energy_start in [("Run A", "100"), ("Run B", "130")]:
        response = client.post(
            "/gratings",
            data={
                "name": name,
                "grating_type": "blazed",
                "period_lpermm": "600",
                "x_resolution_nm": "2.0",
                "z_resolution_nm": "0.5",
                "blaze_angle_deg": "0.75",
                "anti_blaze_angle_deg": "",
                "stack_type": "single_layer",
                "substrate_material": "Si",
                "layer_material": "Au",
                "layer_thickness_nm": "30.0",
            },
        )
        assert response.status_code == 302
        grating_id = GratingStore(tmp_path / "saved_gratings").list()[-1]["id"]
        client.post(
            f"/gratings/{grating_id}/runs",
            data={
                "workflow": "fixed_angle",
                "energy_start_ev": energy_start,
                "energy_stop_ev": str(float(energy_start) + 20.0),
                "energy_points": "3",
                "grazing_angle_deg": "1.5",
                "diffraction_order": "1",
                "fourier_orders": "5",
                "run_x_resolution_nm": "0.75",
                "run_z_resolution_nm": "0.25",
            },
            follow_redirects=True,
        )

    runs = client.get("/").data
    assert b"Run A" in runs
    assert b"Run B" in runs

    plot_index = client.get("/plots")
    assert plot_index.status_code == 200
    assert b"orders_" in plot_index.data

    run_ids = sorted(
        path.parent.name
        for path in (tmp_path / "runs").glob("*/manifest.json")
    )
    assert len(run_ids) == 2

    plot_response = client.post(
        "/plots",
        data=MultiDict(
            [
                ("run_ids", run_ids[0]),
                ("run_ids", run_ids[1]),
                (f"orders_{run_ids[0]}", "1"),
                (f"orders_{run_ids[0]}", "2"),
                (f"orders_{run_ids[1]}", "1"),
            ]
        ),
        follow_redirects=True,
    )

    assert plot_response.status_code == 200
    assert b"Combined plot" in plot_response.data
    assert list((tmp_path / "plots").glob("*/*.png"))
