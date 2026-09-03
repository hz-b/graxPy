# ruff: noqa: D100,D103

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from werkzeug.datastructures import MultiDict

from grax.gratings import BlazedGrating, LaminarGrating
from grax.materials import MaterialSpec
from grax.stacks import MultilayerStack, SingleLayerStack
from grax.web import app as web_app_module
from grax.web.persistence import (
    GratingStore,
    build_grating_from_spec,
    grating_to_spec,
)


def _write_run_fixture(
    base_dir: Path,
    *,
    run_id: str,
    display_name: str,
    grating_name: str = "Demo grating",
    grating_type: str = "laminar",
    workflow: str = "fixed_angle",
    orders: tuple[int, ...] = (1, 2),
    comment: str = "",
) -> None:
    run_dir = base_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "id": run_id,
        "created_at": "2026-06-10T12:00:00",
        "workflow": workflow,
        "grating_id": "grating-1",
        "grating_name": grating_name,
        "grating_spec": {"grating_type": grating_type},
        "display_name": display_name,
        "comment": comment,
        "status": "ok",
        "artifacts": ["all_orders.csv"],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with (run_dir / "all_orders.csv").open("w", encoding="utf-8", newline="") as handle:
        handle.write("case_id,energy_ev,grazing_angle_deg,order,efficiency,diffraction_angle_deg\n")
        for energy in (100.0, 110.0):
            for order in orders:
                stored_order = 0 if int(order) == 0 else -abs(int(order))
                handle.write(
                    f"case-{energy:.1f},{energy:.1f},1.5,{stored_order},"
                    f"{0.05 * order + energy / 1000:.6f},{order * 1.2:.6f}\n"
                )


def _write_checkpoint_fixture(
    base_dir: Path,
    *,
    run_id: str,
    status: str = "paused",
    energies: tuple[float, ...] = (100.0, 110.0),
    order_efficiencies: tuple[float, ...] = (0.11, 0.22),
) -> None:
    """Write one incomplete run fixture backed only by checkpoints."""

    from grax.simulation.models import CaseExecutionResult
    from grax.simulation.serialization import _case_result_to_record

    run_dir = base_dir / "runs" / run_id
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "id": run_id,
        "created_at": "2026-06-10T13:00:00",
        "workflow": "fixed_angle",
        "grating_id": "grating-1",
        "grating_name": "Checkpoint grating",
        "grating_spec": {
            "id": "grating-1",
            "name": "Checkpoint grating",
            "grating_type": "blazed",
            "period_lpermm": 600,
            "x_resolution_nm": 2.0,
            "z_resolution_nm": 0.5,
            "blaze_angle_deg": 0.75,
            "anti_blaze_angle_deg": None,
            "stack": {
                "type": "single_layer",
                "substrate_material": "Si",
                "layer_material": "Au",
                "layer_thickness_nm": 30.0,
                "top_cap_material": None,
                "top_cap_thickness_nm": 0.0,
            },
        },
        "display_name": "Checkpoint run",
        "status": status,
        "artifacts": [],
        "worker_mode": "auto",
        "requested_workers": None,
        "resolved_workers": 2,
        "total_points": 4,
        "run_input": {
            "workflow": "fixed_angle",
            "energy_start_ev": 100.0,
            "energy_stop_ev": 130.0,
            "energy_points": 4,
            "grazing_angle_deg": 1.5,
            "diffraction_order": 1,
            "fourier_orders": 5,
            "max_workers_mode": "auto",
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    checkpoint_lines = []
    for index, (energy, efficiency) in enumerate(zip(energies, order_efficiencies)):
        result = CaseExecutionResult(
            case_id=f"fixed_angle-{index:08d}",
            index=index,
            label=None,
            energy_ev=energy,
            grazing_angle_deg=1.5,
            orders=__import__("numpy").asarray([-1, 0, 1]),
            selected_efficiency=efficiency,
            selected_diffraction_angle_deg=1.2,
            efficiency_all=__import__("numpy").asarray([efficiency, 0.1, 0.0]),
            diffraction_angle_all=__import__("numpy").asarray([1.2, 0.0, -1.2]),
            status="ok",
            case_data={"energy_ev": energy, "grazing_angle_deg": 1.5},
        )
        checkpoint_lines.append(json.dumps(_case_result_to_record(result)))
    (checkpoint_dir / "results.jsonl").write_text("\n".join(checkpoint_lines) + "\n", encoding="utf-8")


def _button_fragment(html: bytes, hook: bytes) -> bytes:
    """Return the short HTML fragment surrounding one button hook."""

    start = html.index(hook)
    return html[start : start + 160]


def _write_saved_png_plot_fixture(base_dir: Path, *, plot_id: str = "plot-legacy") -> None:
    """Write one legacy PNG-backed saved plot manifest."""

    plot_dir = base_dir / "plots" / plot_id
    plot_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "id": plot_id,
        "created_at": "2026-06-18T11:00:00",
        "title": "Legacy combined plot",
        "selected_runs": [{"id": "run-1", "name": "Alpha run", "orders": [1]}],
        "plot_path": "combined.png",
    }
    (plot_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (plot_dir / "combined.png").write_bytes(b"png")


def test_web_install_docs_cover_pypi_and_editable_modes() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    readme_text = (repo_root / "README.md").read_text(encoding="utf-8")
    web_ui_text = (repo_root / "docs" / "installation" / "web-ui.md").read_text(encoding="utf-8")

    for text in (readme_text, web_ui_text):
        assert 'python -m pip install "graxpy[web]"' in text
        assert 'python -m pip install -e ".[web]"' in text
    assert "graxpy.[web]" in web_ui_text


def test_web_runtime_dependency_messages_reference_both_install_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web_app_module, "go", None, raising=False)
    monkeypatch.setattr(web_app_module, "plotly_io", None, raising=False)
    monkeypatch.setattr(web_app_module, "get_plotlyjs", None, raising=False)
    monkeypatch.setattr(web_app_module, "make_subplots", None, raising=False)

    with pytest.raises(RuntimeError) as error_info:
        web_app_module._require_plotly()

    message = str(error_info.value)
    assert 'python -m pip install "graxpy[web]"' in message
    assert 'python -m pip install -e ".[web]"' in message

    app_source = (Path(__file__).resolve().parents[2] / "src" / "grax" / "web" / "app.py").read_text(encoding="utf-8")
    assert 'graxpy[web]' in app_source
    assert '.[web]' in app_source


def test_web_docs_route_and_homepage_link(tmp_path: Path) -> None:
    pytest.importorskip("flask")

    from grax.web.app import create_app

    client = create_app(data_dir=tmp_path).test_client()

    home_response = client.get("/")
    assert home_response.status_code == 200
    assert b'Web docs' in home_response.data
    assert b'href="/docs"' in home_response.data

    grating_response = client.get("/gratings/new")
    assert grating_response.status_code == 200
    assert b'href="/docs"' in grating_response.data

    docs_response = client.get("/docs")
    assert docs_response.status_code == 200
    assert b"Web UI Documentation" in docs_response.data
    assert b".grax-web/" in docs_response.data
    assert b"saved_gratings/" in docs_response.data
    assert b"Materials" in docs_response.data
    assert b"density" in docs_response.data
    assert b"deprecated" in docs_response.data
    assert b"runs/" in docs_response.data
    assert b"plots/" in docs_response.data
    assert b"previews/" in docs_response.data
    assert b"Create and save gratings" in docs_response.data
    assert b"Run simulations" in docs_response.data
    assert b"Compare and plot runs" in docs_response.data
    assert b"How to modify the interface later" in docs_response.data


def test_installed_web_package_includes_templates_and_static_assets() -> None:
    from grax import web as grax_web

    package_root = Path(grax_web.__file__).resolve().parent
    template_root = package_root / "templates"
    static_root = package_root / "static"

    assert (template_root / "index.html").is_file()
    assert (template_root / "web_docs.html").is_file()
    assert (static_root / "web.css").is_file()
    assert (static_root / "web.js").is_file()


def test_saved_grating_round_trips_laminar_multilayer(tmp_path: Path) -> None:
    grating = LaminarGrating(
        period_lpermm=400,
        width_to_period_ratio=0.67,
        depth_nm=14.9,
        left_wall_angle_deg=15.0,
        right_wall_angle_deg=15.0,
        coating_stack=MultilayerStack(
            substrate_material=MaterialSpec("Si", density_g_cm3=2.329),
            material_a=MaterialSpec("Cr", density_g_cm3=7.19),
            material_b=MaterialSpec("C", density_g_cm3=2.2),
            d_period_nm=6.5,
            gamma=0.45,
            n_bilayers=4,
            top_material=MaterialSpec("C", density_g_cm3=2.2),
        ),
        x_resolution_nm=2.0,
        z_resolution_nm=0.5,
    )
    store = GratingStore(tmp_path / "gratings")

    saved = store.save(grating_to_spec(grating, name="Laminar ML"))
    payload = store.load(saved["id"])
    loaded = build_grating_from_spec(payload)

    assert isinstance(loaded, LaminarGrating)
    assert loaded.period_lpermm == 400
    assert loaded.depth_nm == pytest.approx(14.9)
    assert isinstance(loaded.coating_stack, MultilayerStack)
    assert loaded.coating_stack.n_bilayers == 4
    assert loaded.coating_stack.d_period_nm == pytest.approx(6.5)
    assert payload["stack"]["substrate_material"]["name"] == "Si"
    assert payload["stack"]["substrate_material"]["density_g_cm3"] == pytest.approx(2.329)
    assert loaded.coating_stack.substrate_material.name == "Si"
    assert loaded.coating_stack.substrate_material.density_g_cm3 == pytest.approx(2.329)


def test_saved_grating_round_trips_per_layer_roughness(tmp_path: Path) -> None:
    grating = LaminarGrating(
        period_lpermm=400,
        width_to_period_ratio=0.67,
        depth_nm=14.9,
        left_wall_angle_deg=15.0,
        right_wall_angle_deg=15.0,
        coating_stack=SingleLayerStack(
            substrate_material=MaterialSpec("Si", density_g_cm3=2.329),
            layer_material=MaterialSpec("Pt", density_g_cm3=21.46),
            layer_thickness_nm=28.77,
            substrate_roughness_sigma_nm=0.4,
            layer_roughness_sigma_nm=1.2,
        ),
    )
    store = GratingStore(tmp_path / "gratings")

    saved = store.save(grating_to_spec(grating, name="Rough"))
    payload = store.load(saved["id"])

    assert payload["stack"]["substrate_roughness_sigma_nm"] == pytest.approx(0.4)
    assert payload["stack"]["layer_roughness_sigma_nm"] == pytest.approx(1.2)
    assert payload["stack"]["top_cap_roughness_sigma_nm"] is None

    loaded = build_grating_from_spec(payload)
    # interfaces: [substrate boundary, top of layer]
    assert loaded.resolved_stack().interface_roughness_sigmas_bottom_up(0.0) == [0.4, 1.2]


def test_old_grating_spec_without_roughness_still_loads(tmp_path: Path) -> None:
    spec = {
        "name": "Legacy",
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
            "substrate_material": {"name": "Si", "density_g_cm3": 2.329},
            "layer_material": {"name": "Pt", "density_g_cm3": 21.46},
            "layer_thickness_nm": 28.77,
        },
    }

    loaded = build_grating_from_spec(spec)

    assert loaded.resolved_stack().has_per_layer_roughness() is False
    assert loaded.layer_thickness_nm == pytest.approx(28.77)


def test_attach_roughness_sets_and_clears_grating_kind() -> None:
    grating = LaminarGrating(
        coating_stack=SingleLayerStack(
            substrate_material="Si",
            layer_material="Pt",
            layer_thickness_nm=28.77,
            layer_roughness_sigma_nm=1.0,
        )
    )

    web_app_module._attach_roughness(grating, {"roughness_kind": "random-interface"})
    assert grating.roughness is not None
    assert grating.roughness.kind == "random-interface"

    web_app_module._attach_roughness(grating, {"roughness_kind": "debye-waller"})
    assert grating.roughness.kind == "debye-waller"

    web_app_module._attach_roughness(grating, {"roughness_kind": "none"})
    assert grating.roughness is None

    # Missing field defaults to no roughness.
    grating.roughness = object()  # type: ignore[assignment]
    web_app_module._attach_roughness(grating, {})
    assert grating.roughness is None


def test_saved_grating_round_trips_blazed_single_layer(tmp_path: Path) -> None:
    grating = BlazedGrating(
        period_lpermm=600,
        blaze_angle_deg=0.75,
        anti_blaze_angle_deg=5.597,
        substrate_material=MaterialSpec("Si", density_g_cm3=2.329),
        layer_material=MaterialSpec("Au", density_g_cm3=19.3),
        layer_thickness_nm=30.0,
        x_resolution_nm=1.5,
        z_resolution_nm=0.25,
    )
    store = GratingStore(tmp_path / "gratings")

    saved = store.save(grating_to_spec(grating, name="Blazed Au"))
    payload = store.load(saved["id"])
    loaded = build_grating_from_spec(payload)

    assert isinstance(loaded, BlazedGrating)
    assert loaded.period_lpermm == 600
    assert loaded.blaze_angle_deg == pytest.approx(0.75)
    assert loaded.anti_blaze_angle_deg == pytest.approx(5.597)
    assert loaded.layer_thickness_nm == pytest.approx(30.0)
    assert payload["stack"]["layer_material"]["name"] == "Au"
    assert payload["stack"]["layer_material"]["density_g_cm3"] == pytest.approx(19.3)
    assert loaded.layer_material.name == "Au"
    assert loaded.layer_material.density_g_cm3 == pytest.approx(19.3)


def test_saved_grating_round_trips_formula_material_spec(tmp_path: Path) -> None:
    grating = LaminarGrating(
        period_lpermm=400,
        width_to_period_ratio=0.67,
        depth_nm=14.9,
        left_wall_angle_deg=15.0,
        right_wall_angle_deg=15.0,
        substrate_material=MaterialSpec("Si", density_g_cm3=2.329),
        layer_material=MaterialSpec("SiO2", density_g_cm3=2.53),
        layer_thickness_nm=2.0,
        x_resolution_nm=1.5,
        z_resolution_nm=0.25,
    )
    store = GratingStore(tmp_path / "gratings")

    saved = store.save(grating_to_spec(grating, name="Formula coating"))
    payload = store.load(saved["id"])
    loaded = build_grating_from_spec(payload)

    assert payload["stack"]["layer_material"]["name"] == "SiO2"
    assert payload["stack"]["layer_material"]["density_g_cm3"] == pytest.approx(2.53)
    assert loaded.layer_material.name == "SiO2"
    assert loaded.layer_material.density_g_cm3 == pytest.approx(2.53)


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

    assert payload["schema_version"] == 2
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
    assert b"<legend>Substrate</legend>" in response.data
    assert b"<legend>Layer stack</legend>" in response.data
    assert b"<legend>Top cap</legend>" in response.data
    assert b"<legend>Coating</legend>" not in response.data
    assert b'data-grating-section="laminar"' in response.data
    assert b'data-grating-section="blazed"' in response.data
    assert b'data-stack-controls' in response.data
    assert b'data-single-layer-controls' in response.data
    assert b'data-stack-type' in response.data
    assert b"web.js" in response.data
    assert b'name="substrate_material_density_g_cm3"' in response.data
    assert b'name="layer_material_density_g_cm3"' in response.data
    assert b'name="material_a_density_g_cm3"' in response.data
    assert b'name="top_material_density_g_cm3"' in response.data
    assert b'name="substrate_material_density_g_cm3" type="number" step="any" value="2.3296"' in response.data
    assert b'name="layer_material_density_g_cm3" type="number" step="any" value="21.46"' in response.data
    assert b'data-material-select="substrate_material"' in response.data
    assert b'data-material-density="substrate_material"' in response.data
    assert b'list="material-suggestions"' in response.data
    assert b'<datalist id="material-suggestions">' in response.data
    assert b'data-density="19.282"' in response.data
    assert b"Ag" in response.data

    web_js = (Path(__file__).resolve().parents[2] / "src" / "grax" / "web" / "static" / "web.js").read_text(
        encoding="utf-8"
    )
    assert "syncMaterialDensity" in web_js
    assert "[data-material-select]" in web_js


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


def test_flask_app_rejects_unknown_material_names_before_save(tmp_path: Path) -> None:
    pytest.importorskip("flask")

    from grax.web.app import create_app

    app = create_app(data_dir=tmp_path)
    client = app.test_client()
    response = client.post(
        "/gratings",
        data={
            "name": "Broken",
            "grating_type": "blazed",
            "period_lpermm": "600",
            "x_resolution_nm": "2.0",
            "z_resolution_nm": "0.5",
            "blaze_angle_deg": "0.75",
            "anti_blaze_angle_deg": "",
            "stack_type": "single_layer",
            "substrate_material": "Xx",
            "layer_material": "Pt",
            "layer_thickness_nm": "30.0",
        },
    )

    assert response.status_code == 400
    assert b"Unknown element" in response.data


def test_flask_app_accepts_formula_material_names_with_density(tmp_path: Path) -> None:
    pytest.importorskip("flask")

    from grax.web.app import create_app

    app = create_app(data_dir=tmp_path)
    client = app.test_client()
    response = client.post(
        "/gratings",
        data={
            "name": "Formula laminar",
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
            "substrate_material_density_g_cm3": "2.3296",
            "layer_material": "SiO2",
            "layer_material_density_g_cm3": "2.53",
            "layer_thickness_nm": "28.77",
        },
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"Formula laminar" in response.data


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
            "polarization": "p",
            "run_x_resolution_nm": "0.75",
            "run_z_resolution_nm": "0.25",
            "comment": "Commissioning sample",
        },
        follow_redirects=True,
    )

    assert run_response.status_code == 200
    assert b"fixed_angle" in run_response.data
    assert (tmp_path / "runs").exists()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not captured_cases:
        time.sleep(0.02)
    assert captured_cases
    assert captured_cases[0]["x_resolution_nm"] == pytest.approx(0.75)
    assert captured_cases[0]["polarization"] == "p"
    run_id = next((path.parent.name for path in (tmp_path / "runs").glob("*/manifest.json")), None)
    assert run_id is not None
    deadline = time.monotonic() + 2.0
    manifest_path = tmp_path / "runs" / run_id / "manifest.json"
    manifest = {}
    while time.monotonic() < deadline:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") == "completed":
            break
        time.sleep(0.02)
    assert manifest["status"] == "completed"
    assert manifest["display_name"] == "Run grating · fixed_angle"
    assert manifest["comment"] == "Commissioning sample"
    assert manifest["polarization"] == "p"
    assert manifest["run_input"]["polarization"] == "p"
    assert manifest["worker_mode"] == "auto"


def test_grating_form_includes_live_preview_panel(tmp_path: Path) -> None:
    pytest.importorskip("flask")

    from grax.web.app import create_app

    response = create_app(data_dir=tmp_path).test_client().get("/gratings/new")

    assert response.status_code == 200
    assert b'data-grating-preview-form' in response.data
    assert b'data-grating-preview-image' in response.data
    assert b'data-grating-preview-status' in response.data


def test_grating_preview_endpoint_returns_preview_and_validation(tmp_path: Path) -> None:
    pytest.importorskip("flask")

    from grax.web.app import create_app

    client = create_app(data_dir=tmp_path).test_client()

    preview_response = client.post(
        "/_preview/grating",
        data={
            "name": "Preview laminar",
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
            "material_a": "Cr",
            "material_b": "C",
            "d_period_nm": "6.5",
            "gamma": "0.45",
            "n_bilayers": "40",
            "top_material": "C",
            "top_cap_material": "",
            "top_cap_thickness_nm": "0.0",
        },
    )

    assert preview_response.status_code == 200
    preview_payload = preview_response.get_json()
    assert preview_payload["ok"] is True
    assert preview_payload["preview_url"].startswith("/_data/previews/live/")

    invalid_response = client.post(
        "/_preview/grating",
        data={
            "name": "Broken laminar",
            "grating_type": "laminar",
            "period_lpermm": "400",
            "x_resolution_nm": "2.0",
            "z_resolution_nm": "0.5",
            "width_to_period_ratio": "",
            "depth_nm": "14.9",
            "left_wall_angle_deg": "15.0",
            "right_wall_angle_deg": "15.0",
            "stack_type": "single_layer",
            "substrate_material": "Si",
            "layer_material": "Pt",
            "layer_thickness_nm": "28.77",
        },
    )

    assert invalid_response.status_code == 200
    invalid_payload = invalid_response.get_json()
    assert invalid_payload["ok"] is False
    assert invalid_payload["error"]
    assert invalid_payload["preview_url"] is None


def test_plot_page_lists_saved_runs_and_orders(tmp_path: Path) -> None:
    pytest.importorskip("flask")

    from grax.web.app import create_app

    _write_run_fixture(
        tmp_path,
        run_id="run-1",
        display_name="Alpha run",
        grating_name="Alpha grating",
        grating_type="laminar",
        orders=(1, 3),
        comment="Mirror alignment",
    )
    _write_run_fixture(
        tmp_path,
        run_id="run-2",
        display_name="Beta run",
        grating_name="Beta grating",
        grating_type="blazed",
        orders=(2,),
    )

    app = create_app(data_dir=tmp_path)
    response = app.test_client().get("/plots/new")

    assert response.status_code == 200
    assert b"name: Alpha grating" in response.data
    assert b'data-plot-workspace' in response.data
    assert b'data-plot-preview' in response.data
    assert b'data-run-picker' in response.data
    assert b'plot-workspace-layout' in response.data
    assert b'plot-controls-panel' in response.data
    assert b'name="x_axis_type"' in response.data
    assert b'name="y_axis_type"' in response.data
    assert b'data-series-style-list' in response.data
    assert b"name: Alpha grating" in response.data
    assert b"grating type: laminar" in response.data
    assert b"sweep type: fixed_angle" in response.data
    assert b"Mirror alignment" in response.data
    assert b"grating: Alpha grating" not in response.data
    assert b"comment: \xe2\x80\x94" in response.data
    assert b"2026-06-10T12:00:00" in response.data


def test_grating_detail_run_form_exposes_workflow_specific_controls(tmp_path: Path) -> None:
    pytest.importorskip("flask")

    from grax.web.app import create_app

    app = create_app(data_dir=tmp_path)
    client = app.test_client()
    client.post(
        "/gratings",
        data={
            "name": "Workflow grating",
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
    grating_id = GratingStore(tmp_path / "saved_gratings").list()[0]["id"]

    response = client.get(f"/gratings/{grating_id}")

    assert response.status_code == 200
    assert b'data-run-workflow' in response.data
    assert b'data-workflow-fields=' in response.data
    assert b'fixed_angle' in response.data
    assert b'monochromator' in response.data
    assert b'multilayer_theta_search' in response.data
    assert b'parameter_study' in response.data
    assert b'name="comment"' in response.data
    assert b'name="polarization"' in response.data


def test_cases_for_workflow_propagates_polarization_for_web_runs() -> None:
    grating = SimpleNamespace(x_resolution_nm=2.0, z_resolution_nm=0.5)
    energies = web_app_module.np.asarray([100.0, 110.0], dtype=float)
    captured: dict[str, dict[str, object]] = {}

    class FakeSimulation:
        def fixed_angle_cases(self, **kwargs):  # type: ignore[no-untyped-def]
            captured["fixed_angle"] = dict(kwargs)
            return [{"energy_ev": 100.0, "grazing_angle_deg": 1.5}]

        def monochromator_cases(self, **kwargs):  # type: ignore[no-untyped-def]
            captured["monochromator"] = dict(kwargs)
            return [{"energy_ev": 100.0, "grazing_angle_deg": 1.6}]

        def multilayer_theta_search_cases(self, **kwargs):  # type: ignore[no-untyped-def]
            captured["multilayer_theta_search"] = dict(kwargs)
            return [{"energy_ev": 100.0}]

    form = MultiDict(
        {
            "grazing_angle_deg": "1.5",
            "cff": "2.25",
            "polarization": "p",
            "solver": "neviere",
            "run_x_resolution_nm": "0.75",
            "run_z_resolution_nm": "0.25",
        }
    )

    fixed_cases = web_app_module._cases_for_workflow(
        simulation=FakeSimulation(),
        workflow="fixed_angle",
        grating=grating,
        energies=energies,
        form=form,
        diffraction_order=1,
        fourier_orders=5,
    )
    mono_cases = web_app_module._cases_for_workflow(
        simulation=FakeSimulation(),
        workflow="monochromator",
        grating=grating,
        energies=energies,
        form=form,
        diffraction_order=1,
        fourier_orders=5,
    )
    theta_cases = web_app_module._cases_for_workflow(
        simulation=FakeSimulation(),
        workflow="multilayer_theta_search",
        grating=grating,
        energies=energies,
        form=form,
        diffraction_order=1,
        fourier_orders=5,
    )

    assert captured["fixed_angle"]["polarization"] == "p"
    assert captured["monochromator"]["polarization"] == "p"
    assert "polarization" not in captured["multilayer_theta_search"]
    # The theta-search generator resolves solver from the runner default, so the
    # web handler stamps it on the case dict rather than passing it through here.
    assert "solver" not in captured["multilayer_theta_search"]
    assert fixed_cases[0]["polarization"] == "p"
    assert mono_cases[0]["polarization"] == "p"
    assert theta_cases[0]["polarization"] == "p"
    assert fixed_cases[0]["solver"] == "neviere"
    assert mono_cases[0]["solver"] == "neviere"
    assert theta_cases[0]["solver"] == "neviere"


def test_parameter_study_run_uses_selected_polarization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("flask")

    from grax.web.app import create_app

    captured: dict[str, object] = {}

    def fake_run_parameter_study(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return SimpleNamespace()

    def fake_plot_parameter_study(result, output_filename):  # type: ignore[no-untyped-def]
        Path(output_filename).write_text("plot", encoding="utf-8")

    monkeypatch.setattr("grax.parameter_sweep.run_parameter_study", fake_run_parameter_study)
    monkeypatch.setattr("grax.parameter_sweep.plot_parameter_study", fake_plot_parameter_study)

    app = create_app(data_dir=tmp_path)
    client = app.test_client()
    client.post(
        "/gratings",
        data={
            "name": "Parameter grating",
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
    grating_id = GratingStore(tmp_path / "saved_gratings").list()[0]["id"]

    response = client.post(
        f"/gratings/{grating_id}/runs",
        data={
            "workflow": "parameter_study",
            "energy_start_ev": "100",
            "energy_stop_ev": "120",
            "energy_points": "3",
            "grazing_angle_deg": "1.5",
            "diffraction_order": "1",
            "fourier_orders": "5",
            "polarization": "p",
            "run_x_resolution_nm": "0.75",
            "run_z_resolution_nm": "0.25",
        },
        follow_redirects=False,
    )

    assert response.status_code == 302
    run_id = response.headers["Location"].rsplit("/", 1)[-1]
    manifest_path = tmp_path / "runs" / run_id / "manifest.json"

    deadline = time.monotonic() + 2.0
    manifest = {}
    while time.monotonic() < deadline:
        if captured:
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("status") == "completed":
                break
        time.sleep(0.02)

    assert captured["polarization"] == "p"
    assert manifest["polarization"] == "p"
    assert manifest["run_input"]["polarization"] == "p"


def test_load_run_manifest_defaults_missing_polarization_to_s(tmp_path: Path) -> None:
    _write_checkpoint_fixture(tmp_path, run_id="run-checkpoint", status="paused")

    manifest = web_app_module._load_run_manifest(tmp_path, "run-checkpoint")

    assert manifest is not None
    assert manifest["polarization"] == "s"
    assert manifest["run_input"]["polarization"] == "s"


def test_plot_preview_endpoint_returns_live_preview(tmp_path: Path) -> None:
    pytest.importorskip("flask")

    from grax.web.app import create_app

    _write_run_fixture(tmp_path, run_id="run-1", display_name="Alpha run", orders=(0, 1, 3))
    _write_run_fixture(tmp_path, run_id="run-2", display_name="Beta run", orders=(0, 2))

    client = create_app(data_dir=tmp_path).test_client()

    response = client.post(
        "/_preview/plot",
        data={
            "title": "Workspace preview",
            "run_ids": ["run-1", "run-2"],
            "orders_run-1": ["0", "3"],
            "orders_run-2": ["2"],
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["figure_json"]
    assert payload["selected_runs"][0]["orders"] == [0, 3]
    assert len(payload["series_controls"]) == 3
    assert payload["series_controls"][0]["marker_size"] == 3
    assert payload["series_controls"][0]["label"].endswith("· order: 0")
    assert "name: Demo grating" in payload["series_controls"][0]["label"]
    assert "grating type: laminar" in payload["series_controls"][0]["label"]
    assert "sweep type: fixed_angle" in payload["series_controls"][0]["label"]
    assert "grating: Demo grating" not in payload["series_controls"][0]["label"]
    figure = json.loads(payload["figure_json"])
    assert len(figure["data"]) == 3
    assert figure["layout"]["xaxis"]["type"] == "linear"
    assert figure["layout"]["yaxis"]["type"] == "linear"
    assert figure["data"][0]["marker"]["size"] == 3

    invalid_response = client.post("/_preview/plot", data={"title": "Empty"})
    assert invalid_response.status_code == 200
    invalid_payload = invalid_response.get_json()
    assert invalid_payload["ok"] is False
    assert invalid_payload["error"]


def test_plot_preview_applies_series_styles_and_log_axes(tmp_path: Path) -> None:
    pytest.importorskip("flask")

    from grax.web.app import create_app

    _write_run_fixture(tmp_path, run_id="run-1", display_name="Alpha run", orders=(0, 1))
    client = create_app(data_dir=tmp_path).test_client()
    preview_response = client.post(
        "/_preview/plot",
        data={
            "title": "Styled preview",
            "run_ids": ["run-1"],
            "orders_run-1": ["0"],
            "x_axis_type": "log",
            "y_axis_type": "log",
            "color_run_1__order_0": "#123abc",
            "marker_symbol_run_1__order_0": "diamond",
            "marker_size_run_1__order_0": "13",
        },
    )
    preview_payload = preview_response.get_json()
    assert preview_payload["ok"] is True
    figure = json.loads(preview_payload["figure_json"])
    assert figure["layout"]["xaxis"]["type"] == "log"
    assert figure["layout"]["yaxis"]["type"] == "log"
    assert figure["data"][0]["line"]["color"] == "#123abc"
    assert figure["data"][0]["marker"]["symbol"] == "diamond"
    assert figure["data"][0]["marker"]["size"] == 13


def test_saved_plot_persists_interactive_config_and_detail_view(tmp_path: Path) -> None:
    pytest.importorskip("flask")

    from grax.web.app import create_app

    _write_run_fixture(tmp_path, run_id="run-1", display_name="Alpha run", orders=(0, 1))
    _write_run_fixture(tmp_path, run_id="run-2", display_name="Beta run", orders=(2,))
    client = create_app(data_dir=tmp_path).test_client()

    response = client.post(
        "/plots",
        data={
            "title": "Saved interactive plot",
            "run_ids": ["run-1", "run-2"],
            "orders_run-1": ["0", "1"],
            "orders_run-2": ["2"],
            "x_axis_type": "linear",
            "y_axis_type": "log",
            "color_run_1__order_0": "#ff0000",
            "marker_symbol_run_1__order_0": "square",
            "marker_size_run_1__order_0": "11",
        },
        follow_redirects=False,
    )

    assert response.status_code == 302
    location = response.headers["Location"]
    plot_id = location.rsplit("/", 1)[-1]
    manifest = json.loads((tmp_path / "plots" / plot_id / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["plot_config"]["y_axis_type"] == "log"
    assert manifest["plot_config"]["series_styles"]["run-1::order::0"]["color"] == "#ff0000"
    assert manifest["plot_config"]["series_styles"]["run-1::order::0"]["marker_symbol"] == "square"
    assert manifest["plot_config"]["series_styles"]["run-1::order::0"]["marker_size"] == 11
    assert manifest["figure_json"]

    detail_response = client.get(location)
    assert detail_response.status_code == 200
    assert b"interactive Plotly figure" in detail_response.data
    assert b'data-saved-plot-figure' in detail_response.data
    assert b"square" in detail_response.data
    assert b"#ff0000" in detail_response.data


def test_legacy_png_plot_detail_still_renders(tmp_path: Path) -> None:
    pytest.importorskip("flask")

    from grax.web.app import create_app

    _write_saved_png_plot_fixture(tmp_path)

    response = create_app(data_dir=tmp_path).test_client().get("/plots/plot-legacy")

    assert response.status_code == 200
    assert b"Plot image:" in response.data
    assert b"combined.png" in response.data
    assert b'data-saved-plot-figure' not in response.data


def test_saved_run_pages_render_placeholder_for_missing_comment(tmp_path: Path) -> None:
    pytest.importorskip("flask")

    from grax.web.app import create_app

    run_dir = tmp_path / "runs" / "run-legacy"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "id": "run-legacy",
        "created_at": "2026-06-10T12:00:00",
        "workflow": "fixed_angle",
        "grating_id": "grating-1",
        "grating_name": "Legacy grating",
        "grating_spec": {"grating_type": "laminar"},
        "display_name": "Legacy run",
        "status": "ok",
        "artifacts": ["all_orders.csv"],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (run_dir / "all_orders.csv").write_text(
        "case_id,energy_ev,grazing_angle_deg,order,efficiency,diffraction_angle_deg\n"
        "case-100.0,100.0,1.5,-1,0.150000,1.200000\n",
        encoding="utf-8",
    )

    response = create_app(data_dir=tmp_path).test_client().get("/plots/new")

    assert response.status_code == 200
    assert b"name: Legacy grating" in response.data
    assert b"grating type: laminar" in response.data
    assert b"comment: \xe2\x80\x94" in response.data


def test_manage_runs_page_renames_and_deletes_selected_runs(tmp_path: Path) -> None:
    pytest.importorskip("flask")

    from grax.web.app import create_app

    _write_run_fixture(tmp_path, run_id="run-1", display_name="Alpha run", comment="Initial note")
    _write_run_fixture(tmp_path, run_id="run-2", display_name="Beta run")

    app = create_app(data_dir=tmp_path)
    client = app.test_client()

    page = client.get("/runs/manage")
    assert page.status_code == 200
    assert b"data-confirm=\"Delete the selected runs?" in page.data
    assert b'name="comment_run-1"' in page.data
    assert b"Initial note" in page.data

    rename_response = client.post(
        "/runs/manage",
        data={
            "action": "save",
            "display_name_run-1": "Renamed run",
            "comment_run-1": "Updated note",
            "display_name_run-2": "Beta run",
            "comment_run-2": "",
        },
        follow_redirects=True,
    )
    assert rename_response.status_code == 200
    assert b"Renamed run" in rename_response.data
    assert b"Updated note" in rename_response.data
    manifest = json.loads((tmp_path / "runs" / "run-1" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["comment"] == "Updated note"

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


def test_manage_gratings_page_bulk_deletes_selected_gratings(tmp_path: Path) -> None:
    pytest.importorskip("flask")

    from grax.web.app import create_app

    app = create_app(data_dir=tmp_path)
    client = app.test_client()
    for name in ["Alpha grating", "Beta grating"]:
        client.post(
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

    gratings = GratingStore(tmp_path / "saved_gratings").list()
    assert len(gratings) == 2

    page = client.get("/gratings/manage")
    assert page.status_code == 200
    assert b"Manage saved gratings" in page.data
    assert b"Delete selected gratings?" in page.data

    response = client.post(
        "/gratings/manage",
        data={
            "action": "delete",
            "delete_mode": "grating_only",
            "delete_grating_id": [gratings[0]["id"]],
        },
        follow_redirects=True,
    )

    assert response.status_code == 200
    remaining_ids = {grating["id"] for grating in GratingStore(tmp_path / "saved_gratings").list()}
    assert gratings[0]["id"] not in remaining_ids
    assert gratings[1]["id"] in remaining_ids


def test_index_can_switch_active_workspace_root(tmp_path: Path) -> None:
    pytest.importorskip("flask")

    from grax.web.app import create_app

    primary = tmp_path / "primary"
    secondary = tmp_path / "secondary"
    _write_run_fixture(primary, run_id="run-a", display_name="Primary run")
    _write_run_fixture(secondary, run_id="run-b", display_name="Secondary run")

    app = create_app(data_dir=primary)
    client = app.test_client()

    initial = client.get("/")
    assert b"name: Demo grating" in initial.data
    assert b"Secondary run" not in initial.data

    switched = client.post(
        "/workspace",
        data={"workspace_root": str(secondary)},
        follow_redirects=True,
    )

    assert switched.status_code == 200
    assert b"name: Demo grating" in switched.data
    assert b"Primary run" not in switched.data
    assert bytes(str(secondary), "utf-8") in switched.data


def test_workspace_switch_is_rejected_when_active_runs_exist(tmp_path: Path) -> None:
    pytest.importorskip("flask")

    from grax.web.app import ActiveRunState, create_app

    primary = tmp_path / "primary"
    secondary = tmp_path / "secondary"
    app = create_app(data_dir=primary)
    app.extensions["grax_active_runs"] = {
        "run-1": ActiveRunState(
            run_id="run-1",
            workflow="fixed_angle",
            total_points=3,
            worker_mode="auto",
            requested_workers=None,
            resolved_workers=1,
            state="running",
            worker_thread=threading.current_thread(),
        )
    }
    client = app.test_client()

    response = client.post(
        "/workspace",
        data={"workspace_root": str(secondary)},
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"active runs" in response.data.lower()
    assert app.config["GRAx_DATA_DIR"] == primary.resolve()


def test_delete_grating_only_keeps_linked_runs(tmp_path: Path) -> None:
    pytest.importorskip("flask")

    from grax.web.app import create_app

    app = create_app(data_dir=tmp_path)
    client = app.test_client()
    client.post(
        "/gratings",
        data={
            "name": "Delete me",
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
    grating_id = GratingStore(tmp_path / "saved_gratings").list()[0]["id"]
    _write_run_fixture(tmp_path, run_id="run-1", display_name="Run one")
    manifest_path = tmp_path / "runs" / "run-1" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["grating_id"] = grating_id
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    response = client.post(
        f"/gratings/{grating_id}/delete",
        data={"delete_mode": "grating_only"},
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert not (tmp_path / "saved_gratings" / f"{grating_id}.json").exists()
    assert (tmp_path / "runs" / "run-1").exists()


def test_delete_grating_and_runs_removes_linked_runs(tmp_path: Path) -> None:
    pytest.importorskip("flask")

    from grax.web.app import create_app

    app = create_app(data_dir=tmp_path)
    client = app.test_client()
    client.post(
        "/gratings",
        data={
            "name": "Delete cascade",
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
    grating_id = GratingStore(tmp_path / "saved_gratings").list()[0]["id"]
    _write_run_fixture(tmp_path, run_id="run-1", display_name="Run one")
    manifest_path = tmp_path / "runs" / "run-1" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["grating_id"] = grating_id
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    response = client.post(
        f"/gratings/{grating_id}/delete",
        data={"delete_mode": "grating_and_runs"},
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert not (tmp_path / "saved_gratings" / f"{grating_id}.json").exists()
    assert not (tmp_path / "runs" / "run-1").exists()


def test_delete_grating_and_runs_is_blocked_for_active_linked_runs(tmp_path: Path) -> None:
    pytest.importorskip("flask")

    from grax.web.app import ActiveRunState, create_app

    app = create_app(data_dir=tmp_path)
    client = app.test_client()
    client.post(
        "/gratings",
        data={
            "name": "Delete blocked",
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
    grating_id = GratingStore(tmp_path / "saved_gratings").list()[0]["id"]
    _write_run_fixture(tmp_path, run_id="run-1", display_name="Run one")
    manifest_path = tmp_path / "runs" / "run-1" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["grating_id"] = grating_id
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    app.extensions["grax_active_runs"] = {
        "run-1": ActiveRunState(
            run_id="run-1",
            workflow="fixed_angle",
            total_points=3,
            worker_mode="auto",
            requested_workers=None,
            resolved_workers=1,
            state="running",
            worker_thread=threading.current_thread(),
        )
    }

    response = client.post(
        f"/gratings/{grating_id}/delete",
        data={"delete_mode": "grating_and_runs"},
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"active runs" in response.data.lower()
    assert (tmp_path / "saved_gratings" / f"{grating_id}.json").exists()
    assert (tmp_path / "runs" / "run-1").exists()


def test_create_run_redirects_immediately_and_exposes_live_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("flask")

    from grax.simulation.models import CaseExecutionResult
    from grax.web.app import create_app

    def fake_run_cases(self, cases, metadata=None):  # type: ignore[no-untyped-def]
        for index, case in enumerate(cases):
            time.sleep(0.15)
            yield CaseExecutionResult(
                case_id=str(case["case_id"]),
                index=index,
                label=None,
                energy_ev=float(case["energy_ev"]),
                grazing_angle_deg=float(case["grazing_angle_deg"]),
                orders=__import__("numpy").asarray([-1, 0, 1]),
                selected_efficiency=0.2 + index * 0.01,
                selected_diffraction_angle_deg=1.2,
                efficiency_all=__import__("numpy").asarray([0.2, 0.1, 0.0]),
                diffraction_angle_all=__import__("numpy").asarray([1.2, 0.0, -1.2]),
                status="ok",
                case_data={key: value for key, value in case.items() if key != "grating"},
            )

    monkeypatch.setattr("grax.simulation.BatchSimulationRunner.run_cases", fake_run_cases)
    app = create_app(data_dir=tmp_path)
    client = app.test_client()
    client.post(
        "/gratings",
        data={
            "name": "Async run grating",
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
    grating_id = GratingStore(tmp_path / "saved_gratings").list()[0]["id"]

    started = time.monotonic()
    response = client.post(
        f"/gratings/{grating_id}/runs",
        data={
            "workflow": "fixed_angle",
            "energy_start_ev": "100",
            "energy_stop_ev": "140",
            "energy_points": "4",
            "grazing_angle_deg": "1.5",
            "diffraction_order": "1",
            "fourier_orders": "5",
            "max_workers_mode": "manual",
            "max_workers": "2",
        },
        follow_redirects=False,
    )
    elapsed = time.monotonic() - started

    assert response.status_code == 302
    assert elapsed < 0.12
    location = response.headers["Location"]
    run_id = location.rsplit("/", 1)[-1]

    live_response = client.get(f"/runs/{run_id}/status")
    assert live_response.status_code == 200
    live_payload = live_response.get_json()
    assert live_payload["state"] in {"queued", "running", "completed"}
    assert live_payload["worker_mode"] == "manual"
    assert live_payload["requested_workers"] == 2
    assert live_payload["total_points"] == 4
    assert live_payload["remaining_points"] <= 4

    deadline = time.monotonic() + 15.0
    final_payload = live_payload
    while time.monotonic() < deadline:
        final_payload = client.get(f"/runs/{run_id}/status").get_json()
        if final_payload["state"] == "completed":
            break
        time.sleep(0.05)
    assert final_payload["state"] == "completed"
    assert final_payload["completed_points"] == 4
    assert final_payload["remaining_points"] == 0
    assert final_payload["plot_url"]


def test_run_detail_page_exposes_live_monitor_hooks(tmp_path: Path) -> None:
    pytest.importorskip("flask")

    from grax.web.app import create_app

    _write_run_fixture(tmp_path, run_id="run-1", display_name="Alpha run")

    response = create_app(data_dir=tmp_path).test_client().get("/runs/run-1")

    assert response.status_code == 200
    assert b'data-live-run-monitor' in response.data
    assert b'data-run-status-url="/runs/run-1/status"' in response.data
    assert b'data-memory-url="/system/memory"' in response.data


def test_system_memory_endpoint_returns_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("flask")

    from grax.web.app import create_app

    monkeypatch.setattr(
        "grax.web.app.psutil.virtual_memory",
        lambda: SimpleNamespace(total=1000, used=400, available=600, percent=40.0),
    )
    client = create_app(data_dir=tmp_path).test_client()

    response = client.get("/system/memory")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["total_bytes"] == 1000
    assert payload["used_bytes"] == 400
    assert payload["available_bytes"] == 600
    assert payload["percent_used"] == 40.0


def test_memory_status_payload_falls_back_to_proc_meminfo(monkeypatch: pytest.MonkeyPatch) -> None:
    from grax.web.app import _memory_status_payload

    meminfo = "\n".join(
        [
            "MemTotal:       1000 kB",
            "MemAvailable:    250 kB",
            "MemFree:         200 kB",
        ]
    )
    monkeypatch.setattr("grax.web.app.psutil.virtual_memory", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr("pathlib.Path.read_text", lambda self, encoding='utf-8': meminfo)

    payload = _memory_status_payload()

    assert payload["ok"] is True
    assert payload["total_bytes"] == 1000 * 1024
    assert payload["available_bytes"] == 250 * 1024
    assert payload["used_bytes"] == 750 * 1024
    assert payload["percent_used"] == pytest.approx(75.0)


def test_run_status_marks_disk_backed_incomplete_run_as_interrupted(tmp_path: Path) -> None:
    pytest.importorskip("flask")

    from grax.web.app import create_app

    _write_checkpoint_fixture(tmp_path, run_id="run-checkpoint", status="running")

    response = create_app(data_dir=tmp_path).test_client().get("/runs/run-checkpoint/status")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["state"] == "aborted"
    assert payload["can_abort"] is True
    assert payload["checkpoint_completed_points"] == 2
    assert payload["total_points"] == 4


def test_incomplete_run_exposes_available_orders_from_checkpoints(tmp_path: Path) -> None:
    from grax.web.app import _available_orders

    _write_checkpoint_fixture(tmp_path, run_id="run-checkpoint", status="paused")

    orders = _available_orders(tmp_path / "runs" / "run-checkpoint")

    assert orders == [0, 1]


def test_load_order_series_falls_back_to_checkpoint_results(tmp_path: Path) -> None:
    from grax.web.app import _load_order_series

    _write_checkpoint_fixture(tmp_path, run_id="run-checkpoint", status="paused")

    series = _load_order_series(tmp_path / "runs" / "run-checkpoint", order=1, label="Checkpoint run")

    assert series is not None
    assert series["energies"] == [100.0, 110.0]
    assert series["efficiencies"] == [0.11, 0.22]


def test_run_detail_shows_abort_action_for_paused_run(tmp_path: Path) -> None:
    pytest.importorskip("flask")

    from grax.web.app import create_app

    _write_checkpoint_fixture(tmp_path, run_id="run-checkpoint", status="paused")

    response = create_app(data_dir=tmp_path).test_client().get("/runs/run-checkpoint")
    abort_fragment = _button_fragment(response.data, b"data-run-abort-action")

    assert response.status_code == 200
    assert b"Abort run" in response.data
    assert b"Pause run" not in response.data
    assert b"Resume run" not in response.data
    assert b"disabled" not in abort_fragment
    assert b"Machine RAM" in response.data
    assert b"Web RSS" not in response.data
    assert b"Simulation RSS" not in response.data


def test_run_status_prefers_persisted_aborted_state_when_active_entry_is_stale(
    tmp_path: Path,
) -> None:
    pytest.importorskip("flask")

    from grax.web.app import ActiveRunState, create_app

    _write_checkpoint_fixture(tmp_path, run_id="run-checkpoint", status="paused")
    app = create_app(data_dir=tmp_path)
    app.extensions["grax_active_runs"] = {
        "run-checkpoint": ActiveRunState(
            run_id="run-checkpoint",
            workflow="fixed_angle",
            total_points=4,
            worker_mode="auto",
            requested_workers=None,
            resolved_workers=2,
            state="pausing",
            completed_points=1,
        )
    }
    response = app.test_client().get("/runs/run-checkpoint/status")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["state"] == "aborted"
    assert payload["can_abort"] is True
    assert payload["checkpoint_completed_points"] == 2
    assert "memory" not in payload


def test_run_detail_enables_abort_for_paused_run_with_stale_active_entry(tmp_path: Path) -> None:
    pytest.importorskip("flask")

    from grax.web.app import ActiveRunState, create_app

    _write_checkpoint_fixture(tmp_path, run_id="run-checkpoint", status="paused")
    app = create_app(data_dir=tmp_path)
    app.extensions["grax_active_runs"] = {
        "run-checkpoint": ActiveRunState(
            run_id="run-checkpoint",
            workflow="fixed_angle",
            total_points=4,
            worker_mode="auto",
            requested_workers=None,
            resolved_workers=2,
            state="pausing",
            completed_points=1,
        )
    }

    response = app.test_client().get("/runs/run-checkpoint")
    abort_fragment = _button_fragment(response.data, b"data-run-abort-action")

    assert response.status_code == 200
    assert b"Abort run" in response.data
    assert b"data-run-abort-action" in response.data
    assert b"disabled" not in abort_fragment


def test_live_status_returns_without_active_run_lock_deadlock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("flask")

    from grax.web.app import ActiveRunState, create_app

    class FakeProcess:
        def __init__(self, pid: int | None = None) -> None:
            self.pid = pid

        def memory_info(self) -> SimpleNamespace:
            return SimpleNamespace(rss=123)

    monkeypatch.setattr("grax.web.app.psutil.Process", FakeProcess, raising=False)
    app = create_app(data_dir=tmp_path)
    app.extensions["grax_active_runs"] = {
        "run-1": ActiveRunState(
            run_id="run-1",
            workflow="fixed_angle",
            total_points=3,
            worker_mode="auto",
            requested_workers=None,
            resolved_workers=2,
            state="running",
            worker_thread=threading.current_thread(),
        )
    }
    result: dict[str, object] = {}

    def request_status() -> None:
        response = app.test_client().get("/runs/run-1/status")
        result["status_code"] = response.status_code
        result["payload"] = response.get_json()

    thread = threading.Thread(target=request_status, daemon=True)
    thread.start()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert result["status_code"] == 200
    payload = result["payload"]
    assert isinstance(payload, dict)
    assert payload["state"] == "running"
    assert "memory" not in payload


def test_abort_confirmation_page_renders_choices(
    tmp_path: Path,
) -> None:
    pytest.importorskip("flask")

    from grax.web.app import create_app

    _write_checkpoint_fixture(tmp_path, run_id="run-checkpoint", status="paused")

    response = create_app(data_dir=tmp_path).test_client().get("/runs/run-checkpoint/abort")

    assert response.status_code == 200
    assert b"Save partial run" in response.data
    assert b"Delete run" in response.data


def test_abort_route_marks_run_aborted(
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
                orders=__import__("numpy").asarray([-1, 0, 1]),
                selected_efficiency=0.2,
                selected_diffraction_angle_deg=1.2,
                efficiency_all=__import__("numpy").asarray([0.2, 0.1, 0.0]),
                diffraction_angle_all=__import__("numpy").asarray([1.2, 0.0, -1.2]),
                status="ok",
                case_data={key: value for key, value in case.items() if key != "grating"},
            )
            break

    monkeypatch.setattr("grax.simulation.BatchSimulationRunner.run_cases", fake_run_cases)
    app = create_app(data_dir=tmp_path)
    client = app.test_client()
    client.post(
        "/gratings",
        data={
            "name": "Pause me",
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
    grating_id = GratingStore(tmp_path / "saved_gratings").list()[0]["id"]
    response = client.post(
        f"/gratings/{grating_id}/runs",
        data={
            "workflow": "fixed_angle",
            "energy_start_ev": "100",
            "energy_stop_ev": "140",
            "energy_points": "4",
            "grazing_angle_deg": "1.5",
            "diffraction_order": "1",
            "fourier_orders": "5",
        },
        follow_redirects=False,
    )
    run_id = response.headers["Location"].rsplit("/", 1)[-1]

    response = client.post(
        f"/runs/{run_id}/abort",
        data={"disposition": "save"},
        follow_redirects=False,
    )

    assert response.status_code == 302
    deadline = time.monotonic() + 5.0
    final_payload = {}
    while time.monotonic() < deadline:
        final_payload = client.get(f"/runs/{run_id}/status").get_json()
        if final_payload["state"] == "aborted":
            break
        time.sleep(0.05)

    manifest = json.loads((tmp_path / "runs" / run_id / "manifest.json").read_text())
    assert final_payload["state"] == "aborted"
    assert final_payload["can_abort"] is False
    assert manifest["status"] == "aborted"
    assert (tmp_path / "runs" / run_id).exists()


def test_abort_route_can_delete_run_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("flask")

    from grax.web.app import create_app

    _write_checkpoint_fixture(tmp_path, run_id="run-checkpoint", status="paused")
    client = create_app(data_dir=tmp_path).test_client()

    response = client.post(
        "/runs/run-checkpoint/abort",
        data={"disposition": "delete"},
        follow_redirects=False,
    )

    assert response.status_code == 302
    assert not (tmp_path / "runs" / "run-checkpoint").exists()


def test_manage_runs_page_no_longer_shows_resume_buttons(tmp_path: Path) -> None:
    pytest.importorskip("flask")

    from grax.web.app import create_app

    _write_checkpoint_fixture(tmp_path, run_id="run-checkpoint", status="aborted")

    response = create_app(data_dir=tmp_path).test_client().get("/runs/manage")

    assert response.status_code == 200
    assert b"Resume" not in response.data


def test_main_accepts_custom_host_and_port(monkeypatch: pytest.MonkeyPatch) -> None:
    from grax.web import app as web_app

    captured: dict[str, object] = {}

    class DummyApp:
        def run(self, **kwargs):  # type: ignore[no-untyped-def]
            captured.update(kwargs)

    monkeypatch.setattr(web_app, "create_app", lambda: DummyApp())
    monkeypatch.setattr(sys, "argv", ["grax-web", "--host", "0.0.0.0", "--port", "8000"])

    web_app.main()

    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 8000
    assert captured["debug"] is True


def test_main_opens_default_browser_for_local_web_ui(monkeypatch: pytest.MonkeyPatch) -> None:
    from grax.web import app as web_app

    opened: list[str] = []

    class DummyApp:
        def run(self, **kwargs):  # type: ignore[no-untyped-def]
            del kwargs

    class ImmediateTimer:
        def __init__(self, _delay, callback):  # type: ignore[no-untyped-def]
            self.callback = callback
            self.daemon = False

        def start(self):  # type: ignore[no-untyped-def]
            self.callback()

    monkeypatch.setattr(web_app, "create_app", lambda: DummyApp())
    monkeypatch.setattr(web_app.webbrowser, "open", lambda url: opened.append(url))
    monkeypatch.setattr(web_app.threading, "Timer", ImmediateTimer)
    monkeypatch.setattr(sys, "argv", ["grax-web"])
    monkeypatch.setenv("WERKZEUG_RUN_MAIN", "true")

    web_app.main()

    assert opened == ["http://127.0.0.1:5050"]


def test_main_opens_localhost_when_bound_to_all_interfaces(monkeypatch: pytest.MonkeyPatch) -> None:
    from grax.web import app as web_app

    opened: list[str] = []

    class DummyApp:
        def run(self, **kwargs):  # type: ignore[no-untyped-def]
            del kwargs

    class ImmediateTimer:
        def __init__(self, _delay, callback):  # type: ignore[no-untyped-def]
            self.callback = callback
            self.daemon = False

        def start(self):  # type: ignore[no-untyped-def]
            self.callback()

    monkeypatch.setattr(web_app, "create_app", lambda: DummyApp())
    monkeypatch.setattr(web_app.webbrowser, "open", lambda url: opened.append(url))
    monkeypatch.setattr(web_app.threading, "Timer", ImmediateTimer)
    monkeypatch.setattr(sys, "argv", ["grax-web", "--host", "0.0.0.0", "--port", "8000"])
    monkeypatch.setenv("WERKZEUG_RUN_MAIN", "true")

    web_app.main()

    assert opened == ["http://127.0.0.1:8000"]


def test_base_template_renders_global_attribution_links(tmp_path: Path) -> None:
    pytest.importorskip("flask")

    from grax.web.app import create_app

    response = create_app(data_dir=tmp_path).test_client().get("/")

    assert response.status_code == 200
    assert b"Helmholtz-Zentrum Berlin" in response.data
    assert b"Simone Vadilonga" in response.data
    assert b"github.com/hz-b/graxPy" in response.data
    assert b"graxpy.readthedocs.io/en/latest/" in response.data


def test_results_sorted_for_live_plot_orders_by_energy() -> None:
    from grax.simulation.models import CaseExecutionResult
    from grax.web.app import _results_sorted_for_live_plot

    results = [
        CaseExecutionResult(
            case_id="case-2",
            index=1,
            label=None,
            energy_ev=200.0,
            grazing_angle_deg=1.5,
            orders=__import__("numpy").asarray([-1, 0, 1]),
            selected_efficiency=0.2,
            selected_diffraction_angle_deg=1.0,
            efficiency_all=__import__("numpy").asarray([0.2, 0.1, 0.0]),
            diffraction_angle_all=__import__("numpy").asarray([1.0, 0.0, -1.0]),
            status="ok",
            case_data={},
        ),
        CaseExecutionResult(
            case_id="case-1",
            index=0,
            label=None,
            energy_ev=100.0,
            grazing_angle_deg=1.5,
            orders=__import__("numpy").asarray([-1, 0, 1]),
            selected_efficiency=0.1,
            selected_diffraction_angle_deg=1.0,
            efficiency_all=__import__("numpy").asarray([0.1, 0.1, 0.0]),
            diffraction_angle_all=__import__("numpy").asarray([1.0, 0.0, -1.0]),
            status="ok",
            case_data={},
        ),
    ]

    ordered = _results_sorted_for_live_plot(results, x_key="energy_ev")

    assert [result.energy_ev for result in ordered] == [100.0, 200.0]


def test_publish_live_progress_is_throttled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from grax.simulation.models import CaseExecutionResult
    from grax.web.app import ActiveRunState, _publish_live_progress_snapshot

    state = ActiveRunState(
        run_id="run-1",
        workflow="fixed_angle",
        total_points=3,
        worker_mode="auto",
        requested_workers=None,
        resolved_workers=1,
    )
    results = [
        CaseExecutionResult(
            case_id="case-1",
            index=0,
            label=None,
            energy_ev=100.0,
            grazing_angle_deg=1.5,
            orders=__import__("numpy").asarray([-1, 0, 1]),
            selected_efficiency=0.1,
            selected_diffraction_angle_deg=1.0,
            efficiency_all=__import__("numpy").asarray([0.1, 0.1, 0.0]),
            diffraction_angle_all=__import__("numpy").asarray([1.0, 0.0, -1.0]),
            status="ok",
            case_data={},
        )
    ]
    published = []

    def fake_plot_order_subset(results_arg, output_filename, *, diffraction_orders, title):  # type: ignore[no-untyped-def]
        published.append([result.energy_ev for result in results_arg])
        Path(output_filename).write_bytes(b"png")

    monkeypatch.setattr("grax.simulation.plot_order_subset", fake_plot_order_subset)

    first = _publish_live_progress_snapshot(
        state=state,
        results=results,
        output_path=tmp_path / "live_progress.png",
        diffraction_order=1,
        title="Demo",
        now=10.0,
        min_interval_seconds=1.5,
    )
    first_token = state.plot_token
    second = _publish_live_progress_snapshot(
        state=state,
        results=results,
        output_path=tmp_path / "live_progress.png",
        diffraction_order=1,
        title="Demo",
        now=10.5,
        min_interval_seconds=1.5,
    )

    assert first is True
    assert second is False
    assert state.plot_token == first_token
    assert published == [[100.0]]


def test_load_order_series_uses_selected_diffraction_order_convention(tmp_path: Path) -> None:
    from grax.web.app import _load_order_series

    run_dir = tmp_path / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "all_orders.csv").write_text(
        "\n".join(
            [
                "case_id,energy_ev,grazing_angle_deg,order,efficiency,diffraction_angle_deg",
                "case-1,100.0,1.5,-1,0.11,1.2",
                "case-1,100.0,1.5,1,0.00,-1.2",
                "case-2,200.0,1.5,-1,0.22,1.3",
                "case-2,200.0,1.5,1,0.00,-1.3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    series = _load_order_series(run_dir, order=1, label="Demo")

    assert series is not None
    assert series["order"] == 1
    assert series["energies"] == [100.0, 200.0]
    assert series["efficiencies"] == [0.11, 0.22]


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

    deadline = time.monotonic() + 5.0
    manifests = []
    while time.monotonic() < deadline:
        manifests = [
            json.loads(path.read_text())
            for path in (tmp_path / "runs").glob("*/manifest.json")
        ]
        if manifests and all(manifest.get("status") == "completed" for manifest in manifests):
            break
        time.sleep(0.05)
    assert manifests
    assert all(manifest.get("status") == "completed" for manifest in manifests)

    runs = client.get("/").data
    assert b"Run A" in runs
    assert b"Run B" in runs

    plot_index = client.get("/plots/new")
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
    manifests = list((tmp_path / "plots").glob("*/manifest.json"))
    assert manifests
    saved_manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert saved_manifest["figure_json"]
    assert saved_manifest["plot_config"]["series_styles"]


def test_parameter_study_run_uses_selected_solver(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify the run form's solver choice reaches the parameter-study call."""

    pytest.importorskip("flask")

    from grax.web.app import create_app

    captured: dict[str, object] = {}

    def fake_run_parameter_study(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return SimpleNamespace()

    def fake_plot_parameter_study(result, output_filename):  # type: ignore[no-untyped-def]
        Path(output_filename).write_text("plot", encoding="utf-8")

    monkeypatch.setattr("grax.parameter_sweep.run_parameter_study", fake_run_parameter_study)
    monkeypatch.setattr("grax.parameter_sweep.plot_parameter_study", fake_plot_parameter_study)

    app = create_app(data_dir=tmp_path)
    client = app.test_client()
    client.post(
        "/gratings",
        data={
            "name": "Solver grating",
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
    grating_id = GratingStore(tmp_path / "saved_gratings").list()[0]["id"]

    response = client.post(
        f"/gratings/{grating_id}/runs",
        data={
            "workflow": "parameter_study",
            "energy_start_ev": "100",
            "energy_stop_ev": "120",
            "energy_points": "3",
            "grazing_angle_deg": "1.5",
            "diffraction_order": "1",
            "fourier_orders": "5",
            "polarization": "p",
            "solver": "neviere",
            "run_x_resolution_nm": "0.75",
            "run_z_resolution_nm": "0.25",
        },
        follow_redirects=False,
    )

    assert response.status_code == 302
    run_id = response.headers["Location"].rsplit("/", 1)[-1]
    manifest_path = tmp_path / "runs" / run_id / "manifest.json"

    deadline = time.monotonic() + 2.0
    manifest: dict[str, object] = {}
    while time.monotonic() < deadline:
        if captured:
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("status") == "completed":
                break
        time.sleep(0.02)

    assert captured["solver"] == "neviere"
    assert manifest.get("solver") == "neviere"
    assert manifest.get("run_input", {}).get("solver") == "neviere"


def test_run_form_rejects_an_unknown_solver(tmp_path: Path) -> None:
    """Verify an invalid solver name is refused rather than silently defaulted."""

    pytest.importorskip("flask")

    from grax.web.app import _normalized_solver

    assert _normalized_solver("neviere") == "neviere"
    assert _normalized_solver(None) == "rcwa"
    assert _normalized_solver("  RCWA  ") == "rcwa"
    with pytest.raises(ValueError, match="solver must be"):
        _normalized_solver("differential")
