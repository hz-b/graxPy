# ruff: noqa: D100,D103

from __future__ import annotations

import json
import time
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _material_form() -> dict[str, str]:
    return {
        "material_a_name": "Ru",
        "material_a_density": "12.1",
        "material_b_name": "C",
        "material_b_density": "2.52",
        "substrate_material_name": "Si",
        "substrate_material_density": "2.33",
    }


def _install_fake_stages(
    monkeypatch: pytest.MonkeyPatch, *, blaze_loops: int = 0
) -> dict[str, int]:
    """Replace the three study runners with fast fakes that write minimal outputs."""

    import grax.multilayer_optimization as mlo

    calls = {"d_spacing": 0, "gamma": 0, "blaze": 0}

    def _make(stage: str, **suggested: float):
        def runner(config, *, progress_callback=None, should_continue=None):  # type: ignore[no-untyped-def]
            calls[stage] += 1
            config.plot_dir.mkdir(parents=True, exist_ok=True)
            plot_name = {
                "d_spacing": "0_d_spacing_study.png",
                "gamma": "1_gamma_study.png",
                "blaze": "2_blaze_study.png",
            }[stage]
            (config.plot_dir / plot_name).write_bytes(b"png")
            csv_dir = {
                "d_spacing": config.d_spacing_results_dir,
                "gamma": config.gamma_results_dir,
                "blaze": config.blaze_results_dir,
            }[stage]
            csv_dir.mkdir(parents=True, exist_ok=True)
            csv_name = {
                "d_spacing": "d_spacing_study.csv",
                "gamma": "gamma_study.csv",
                "blaze": "blaze_study.csv",
            }[stage]
            (csv_dir / csv_name).write_text("energy_ev,value\n9000,1\n", encoding="utf-8")
            state = {}
            if config.state_path.exists():
                state = json.loads(config.state_path.read_text(encoding="utf-8"))
            state.update(suggested)
            config.state_path.write_text(json.dumps(state), encoding="utf-8")
            aborted = False
            iterations = blaze_loops if stage == "blaze" else 3
            for index in range(iterations):
                if should_continue is not None and not should_continue():
                    aborted = True
                    break
                if progress_callback is not None:
                    progress_callback(
                        mlo.StageProgress(
                            stage=stage, completed=index, total=iterations, current_label="x"
                        )
                    )
            return types.SimpleNamespace(aborted=aborted, **suggested)

        return runner

    monkeypatch.setattr(
        mlo,
        "run_d_spacing_study",
        _make(
            "d_spacing",
            geometry_grazing_angle_deg=1.0,
            geometry_d_nm=3.8,
            d_suggested_nm=3.8,
            d_suggested_peak_rp=0.8,
            d_reflectivity_best_nm=4.7,
            d_reflectivity_best_peak_rp=0.82,
        ),
    )
    monkeypatch.setattr(
        mlo,
        "run_gamma_study",
        _make("gamma", d_spacing_nm=3.8, gamma_suggested=0.5, gamma_suggested_peak_rp=0.6),
    )
    monkeypatch.setattr(
        mlo,
        "run_blaze_study",
        _make(
            "blaze",
            d_spacing_nm=3.8,
            gamma=0.5,
            blaze_suggested_deg=1.1,
            blaze_suggested_efficiency=0.3,
        ),
    )
    return calls


def _wait_for_stage(store: object, study_id: str, stage: str, *, timeout: float = 4.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        manifest = store.load(study_id)  # type: ignore[attr-defined]
        if manifest["stages"][stage]["status"] in {"completed", "failed", "aborted", "stale"}:
            return manifest
        time.sleep(0.02)
    return store.load(study_id)  # type: ignore[attr-defined]


def _create_study(client: object, extra: dict[str, str] | None = None) -> str:
    data = {"display_name": "Web test", "d_spacing_points": "3", **_material_form()}
    if extra:
        data.update(extra)
    response = client.post("/multilayer", data=data)  # type: ignore[attr-defined]
    assert response.status_code == 302
    return response.headers["Location"].rsplit("/", 1)[-1]


def test_multilayer_index_and_nav(tmp_path: Path) -> None:
    pytest.importorskip("flask")
    from grax.web.app import create_app

    client = create_app(data_dir=tmp_path).test_client()
    response = client.get("/multilayer")
    assert response.status_code == 200
    assert b"Multilayer optimization studies" in response.data
    assert b'href="/multilayer"' in client.get("/").data


def test_new_study_form_has_curated_and_advanced_sections(tmp_path: Path) -> None:
    pytest.importorskip("flask")
    from grax.web.app import create_app

    response = create_app(data_dir=tmp_path).test_client().get("/multilayer/new")
    assert response.status_code == 200
    assert b"<summary>Advanced</summary>" in response.data
    assert b'name="target_energy_ev"' in response.data
    assert b'name="material_a_name"' in response.data


def test_material_fields_use_the_full_catalog_with_density_hints(tmp_path: Path) -> None:
    pytest.importorskip("flask")
    from grax.materials import available_material_symbols
    from grax.web.app import create_app

    html = create_app(data_dir=tmp_path).test_client().get("/multilayer/new").get_data(as_text=True)
    # The datalist lists every catalog symbol, each carrying a density hint.
    assert html.count('<option value="') >= len(available_material_symbols())
    assert '<option value="Au" data-density="19.282">' in html
    # Name and density inputs are paired for the auto-fill script.
    for key in ("material_a", "material_b", "substrate_material"):
        assert f'data-material-select="{key}"' in html
        assert f'data-material-density="{key}"' in html


def test_create_study_writes_manifest(tmp_path: Path) -> None:
    pytest.importorskip("flask")
    from grax.web.app import create_app
    from grax.web.multilayer_studies import MultilayerStudyStore

    client = create_app(data_dir=tmp_path).test_client()
    study_id = _create_study(client)
    manifest = MultilayerStudyStore(tmp_path / "multilayer_studies").load(study_id)
    assert {s: manifest["stages"][s]["status"] for s in ("d_spacing", "gamma", "blaze")} == {
        "d_spacing": "not_run",
        "gamma": "not_run",
        "blaze": "not_run",
    }
    assert manifest["config"]["material_a"] == ["Ru", 12.1]
    assert client.get(f"/multilayer/{study_id}").status_code == 200


def test_run_stage_completes_and_records_suggestion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("flask")
    from grax.web.app import create_app
    from grax.web.multilayer_studies import MultilayerStudyStore

    _install_fake_stages(monkeypatch)
    client = create_app(data_dir=tmp_path).test_client()
    store = MultilayerStudyStore(tmp_path / "multilayer_studies")
    study_id = _create_study(client)

    assert client.post(f"/multilayer/{study_id}/stages/d_spacing/run", data={}).status_code == 302
    manifest = _wait_for_stage(store, study_id, "d_spacing")
    assert manifest["stages"]["d_spacing"]["status"] == "completed"
    assert manifest["stages"]["d_spacing"]["suggested"]["d_suggested_nm"] == 3.8
    assert manifest["stages"]["gamma"]["status"] == "not_run"
    study_dir = tmp_path / "multilayer_studies" / study_id
    assert (study_dir / "plot" / "0_d_spacing_study.png").is_file()
    assert (study_dir / "0_d_spacing" / "d_spacing_study.csv").is_file()

    status = client.get(f"/multilayer/{study_id}/stages/d_spacing/status").get_json()
    assert set(status) >= {"state", "completed_points", "total_points", "plot_url", "can_abort"}


def test_rerunning_upstream_marks_downstream_stale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("flask")
    from grax.web.app import create_app
    from grax.web.multilayer_studies import MultilayerStudyStore

    _install_fake_stages(monkeypatch)
    client = create_app(data_dir=tmp_path).test_client()
    store = MultilayerStudyStore(tmp_path / "multilayer_studies")
    study_id = _create_study(client)

    client.post(f"/multilayer/{study_id}/stages/d_spacing/run", data={})
    _wait_for_stage(store, study_id, "d_spacing")
    client.post(f"/multilayer/{study_id}/stages/gamma/run", data={"d_spacing_nm": "auto"})
    _wait_for_stage(store, study_id, "gamma")
    assert store.load(study_id)["stages"]["gamma"]["status"] == "completed"

    client.post(f"/multilayer/{study_id}/stages/d_spacing/run", data={})
    _wait_for_stage(store, study_id, "d_spacing")
    assert store.load(study_id)["stages"]["gamma"]["status"] == "stale"
    # the stale stage keeps its files
    assert (tmp_path / "multilayer_studies" / study_id / "1_gamma" / "gamma_study.csv").is_file()


def test_reset_stage_clears_outputs_and_state_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("flask")
    from grax.web.app import create_app
    from grax.web.multilayer_studies import MultilayerStudyStore

    _install_fake_stages(monkeypatch)
    client = create_app(data_dir=tmp_path).test_client()
    store = MultilayerStudyStore(tmp_path / "multilayer_studies")
    study_id = _create_study(client)

    client.post(f"/multilayer/{study_id}/stages/d_spacing/run", data={})
    _wait_for_stage(store, study_id, "d_spacing")
    client.post(f"/multilayer/{study_id}/stages/gamma/run", data={"d_spacing_nm": "auto"})
    _wait_for_stage(store, study_id, "gamma")

    assert client.post(f"/multilayer/{study_id}/stages/gamma/reset", data={}).status_code == 302
    study_dir = tmp_path / "multilayer_studies" / study_id
    assert not (study_dir / "1_gamma").exists()
    assert store.load(study_id)["stages"]["gamma"]["status"] == "not_run"
    state = json.loads((study_dir / "optimization_state.json").read_text(encoding="utf-8"))
    assert "gamma_suggested" not in state
    assert "d_suggested_nm" in state  # d-spacing keys untouched


def test_abort_stage_discard_removes_partial_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("flask")
    from grax.web.app import create_app
    from grax.web.multilayer_studies import MultilayerStudyStore

    _install_fake_stages(monkeypatch, blaze_loops=200)
    client = create_app(data_dir=tmp_path).test_client()
    store = MultilayerStudyStore(tmp_path / "multilayer_studies")
    study_id = _create_study(client)

    client.post(f"/multilayer/{study_id}/stages/blaze/run", data={"d_spacing_nm": "3.8"})
    # give the worker a moment to enter its loop, then discard
    time.sleep(0.1)
    assert client.post(
        f"/multilayer/{study_id}/stages/blaze/abort", data={"disposition": "discard"}
    ).status_code == 302
    manifest = store.load(study_id)
    assert manifest["stages"]["blaze"]["status"] == "not_run"
    assert not (tmp_path / "multilayer_studies" / study_id / "2_blaze").exists()


def test_delete_study_removes_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("flask")
    from grax.web.app import create_app
    from grax.web.multilayer_studies import MultilayerStudyStore

    _install_fake_stages(monkeypatch)
    client = create_app(data_dir=tmp_path).test_client()
    store = MultilayerStudyStore(tmp_path / "multilayer_studies")
    study_id = _create_study(client)
    client.post(f"/multilayer/{study_id}/stages/d_spacing/run", data={})
    _wait_for_stage(store, study_id, "d_spacing")

    assert client.post(f"/multilayer/{study_id}/delete", data={}).status_code == 302
    assert not (tmp_path / "multilayer_studies" / study_id).exists()
    assert store.list() == []


def test_study_id_path_traversal_is_rejected(tmp_path: Path) -> None:
    pytest.importorskip("flask")
    from grax.web.app import create_app

    client = create_app(data_dir=tmp_path).test_client()
    assert client.get("/multilayer/..%2f..%2fetc").status_code == 404
