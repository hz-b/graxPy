# ruff: noqa: D100,D103
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from grax.web.multilayer_design_studies import MultilayerDesignStudyStore
from grax.web.multilayer_diagnostics import build_plot, discover_plots

pytestmark = pytest.mark.unit

CONFIG = {
    "material_a": ["Ru", 12.1],
    "material_b": ["C", 2.52],
    "target_energy_ev": 9000.0,
    "diffraction_order": 2,
    "polarization": "p",
}


def _write_theta_scan(directory: Path, energy_tag: str = "4895eV") -> None:
    scans = directory / "theta_scans"
    scans.mkdir(parents=True)
    pd.DataFrame(
        {
            "scan_type": ["rough", "rough", "precise", "precise", "precise"],
            "grazing_angle_deg": [1.0, 1.2, 1.1, 1.15, 1.2],
            "selected_efficiency": [0.1, 0.2, 0.3, 0.4, 0.3],
            "is_selected_peak": [0, 0, 0, 1, 0],
        }
    ).to_csv(scans / f"theta_scan_{energy_tag}.csv", index=False)


def _make_study_tree(root: Path) -> Path:
    run = root / "survey" / "runs" / "d3.500nm" / "blaze0.960deg"
    run.mkdir(parents=True)
    _write_theta_scan(run)
    pd.DataFrame(
        {
            "d_nm": [3.0, 3.0, 3.5, 3.5],
            "blaze_deg": [0.6, 0.9, 0.6, 0.9],
            "peak_efficiency": [0.1, 0.3, 0.2, 0.4],
        }
    ).to_csv(root / "survey" / "survey.csv", index=False)
    return run


def test_discover_plots_offers_theta_scans_and_survey_plots(tmp_path: Path) -> None:
    run = _make_study_tree(tmp_path)

    root_ids = {item["id"] for item in discover_plots(tmp_path, tmp_path, CONFIG)}
    run_ids = {item["id"] for item in discover_plots(tmp_path, run, CONFIG)}

    assert {"survey-optimal", "survey-maximum", "survey-heatmap", "theta-all"} <= root_ids
    assert run_ids == {"theta-all"}


def test_theta_plot_separates_scan_kinds_and_marks_the_selected_peak(tmp_path: Path) -> None:
    run = _make_study_tree(tmp_path)
    descriptor = discover_plots(tmp_path, run, CONFIG)[0]

    traces = build_plot(descriptor, tmp_path, CONFIG)["traces"]

    assert sorted(trace["role"] for trace in traces) == ["legend", "peak", "scan", "scan"]
    assert {trace["group"] for trace in traces} == {0}
    assert sorted(t["scan_class"] for t in traces if t["role"] == "scan") == ["precise", "rough"]
    star = next(trace for trace in traces if trace["name"].endswith("selected peak"))
    assert star["x"] == [1.15]


@pytest.mark.parametrize("plot_id", ["survey-optimal", "survey-maximum", "survey-heatmap"])
def test_survey_plots_build(tmp_path: Path, plot_id: str) -> None:
    _make_study_tree(tmp_path)
    descriptor = next(
        item for item in discover_plots(tmp_path, tmp_path, CONFIG) if item["id"] == plot_id
    )

    assert build_plot(descriptor, tmp_path, CONFIG)["traces"]


def _client_with_study(tmp_path: Path):  # noqa: ANN202
    pytest.importorskip("flask")
    from grax.web.app import create_app

    app = create_app(data_dir=tmp_path)
    store = MultilayerDesignStudyStore(tmp_path / "multilayer_designs")
    study = store.create(display_name="Diagnostics study", config=CONFIG)
    study_id = str(study["id"])
    _make_study_tree(store.study_dir(study_id))

    def complete(manifest):
        manifest["stages"]["survey"]["status"] = "completed"

    store.update(study_id, complete)
    return app.test_client(), study_id


def test_diagnostics_page_lists_folders_and_plots(tmp_path: Path) -> None:
    client, study_id = _client_with_study(tmp_path)

    detail = client.get(f"/multilayer-design/{study_id}").get_data(as_text=True)
    root = client.get(f"/multilayer-design/{study_id}/diagnostics")
    run = client.get(
        f"/multilayer-design/{study_id}/diagnostics",
        query_string={"path": "survey/runs/d3.500nm/blaze0.960deg", "plot_id": "theta-all"},
    )

    assert f"/multilayer-design/{study_id}/diagnostics" in detail
    assert root.status_code == 200 and b"survey" in root.data
    assert run.status_code == 200 and b"diagnostic-plot-data" in run.data


@pytest.mark.parametrize("path", ["../../etc", "/etc", "survey/missing"])
def test_diagnostics_never_leaves_the_study_folder(tmp_path: Path, path: str) -> None:
    client, study_id = _client_with_study(tmp_path)

    response = client.get(f"/multilayer-design/{study_id}/diagnostics", query_string={"path": path})

    assert response.status_code == 200
    assert b"Study root" in response.data and b"etc" not in response.data.split(b"<h2>")[1][:40]


def test_diagnostics_unknown_study_and_plot(tmp_path: Path) -> None:
    client, study_id = _client_with_study(tmp_path)

    assert client.get("/multilayer-design/nope/diagnostics").status_code == 404
    page = client.get(f"/multilayer-design/{study_id}/diagnostics?plot_id=bogus")
    assert b"not available in this folder" in page.data
