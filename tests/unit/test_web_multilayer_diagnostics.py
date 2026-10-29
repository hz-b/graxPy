# ruff: noqa: D100,D103
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from grax.web.multilayer_design_studies import MultilayerDesignStudyStore
from grax.web.multilayer_diagnostics import build_plot, discover_plots, energy_scan_scans, survey_cell_scans

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


def test_theta_plot_adds_a_dashed_fit_trace_when_the_scan_file_stores_one(tmp_path: Path) -> None:
    run = _make_study_tree(tmp_path)
    scan_file = next((run / "theta_scans").glob("theta_scan_*.csv"))
    fit = pd.DataFrame({"scan_type": ["fit", "fit"], "grazing_angle_deg": [1.1, 1.2],
                        "selected_efficiency": [0.35, 0.38], "is_selected_peak": [0, 0]})
    pd.concat([pd.read_csv(scan_file), fit]).to_csv(scan_file, index=False)
    descriptor = discover_plots(tmp_path, run, CONFIG)[0]

    traces = build_plot(descriptor, tmp_path, CONFIG)["traces"]

    assert not any(trace["name"].endswith("maximum") for trace in traces)  # the star already marks it
    fit_trace = next(trace for trace in traces if trace.get("scan_class") == "fit")
    assert fit_trace["role"] == "scan"
    assert fit_trace["line"]["color"] == "red"
    assert traces.index(fit_trace) > max(i for i, t in enumerate(traces) if t["scan_class"] in {"rough", "precise"} and t["role"] == "scan")
    assert fit_trace["x"] == [1.1, 1.2]


def test_survey_cell_scans_picks_the_clicked_newest_or_best_cell(tmp_path: Path) -> None:
    run = _make_study_tree(tmp_path)  # d = 3.5, blaze = 0.96
    other = tmp_path / "survey" / "runs" / "d3.000nm" / "blaze0.900deg"
    other.mkdir(parents=True)
    _write_theta_scan(other)
    best_cell = tmp_path / "survey" / "runs" / "d3.500nm" / "blaze0.900deg"
    best_cell.mkdir(parents=True)
    _write_theta_scan(best_cell)
    import os
    for stamp, folder in enumerate((run, best_cell, other), start=1):
        os.utime(next((folder / "theta_scans").glob("*.csv")), (stamp, stamp))

    clicked = survey_cell_scans(tmp_path, 3.5, 0.96)
    newest = survey_cell_scans(tmp_path, latest=True)
    best = survey_cell_scans(tmp_path)

    assert (clicked["d_nm"], clicked["blaze_deg"]) == (3.5, 0.96)
    assert (newest["d_nm"], newest["blaze_deg"]) == (3.0, 0.9)
    assert (best["d_nm"], best["blaze_deg"]) == (3.5, 0.9)  # survey.csv best row
    assert {"role", "group", "scan_class"}.isdisjoint(clicked["traces"][0])
    assert survey_cell_scans(tmp_path, 9.0, 9.0)["traces"] == []


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


def test_theta_plot_also_marks_the_sampled_maximum_when_the_fit_picks_another_peak(tmp_path: Path) -> None:
    run = _make_study_tree(tmp_path)
    scan_file = next((run / "theta_scans").glob("theta_scan_*.csv"))
    table = pd.read_csv(scan_file)
    table["is_selected_peak"] = 0
    fit_peak = pd.DataFrame({"scan_type": ["peak"], "grazing_angle_deg": [1.16],
                             "selected_efficiency": [0.41], "is_selected_peak": [1]})
    pd.concat([table, fit_peak]).to_csv(scan_file, index=False)
    descriptor = discover_plots(tmp_path, run, CONFIG)[0]

    traces = build_plot(descriptor, tmp_path, CONFIG)["traces"]

    star = next(trace for trace in traces if trace["name"].endswith("selected peak"))
    maximum = next(trace for trace in traces if trace["name"].endswith("maximum"))
    assert star["x"] == [1.16]
    assert maximum["x"] == [1.15]
    assert [t["scan_class"] for t in traces if t["role"] == "scan"].count("precise") == 1


def test_energy_scan_scans_picks_the_clicked_newest_or_first_design_best_energy(tmp_path: Path) -> None:
    import os
    for name, energy, stamp in (("d3.500nm_blaze0.960deg", "4895eV", 1), ("d4.000nm_blaze0.900deg", "5000eV", 2)):
        design = tmp_path / "energy_scan" / name
        _write_theta_scan(design, energy)
        os.utime(design / "theta_scans" / f"theta_scan_{energy}.csv", (stamp, stamp))
        pd.DataFrame({"energy_ev": [4895.0, 5000.0], "selected_efficiency": [0.2, 0.1]}).to_csv(
            design / "multilayer_theta_search_summary.csv", index=False
        )

    clicked = energy_scan_scans(tmp_path, 4.0, 0.9, 5000.0)
    newest = energy_scan_scans(tmp_path, latest=True)
    first = energy_scan_scans(tmp_path)

    assert (clicked["d_nm"], clicked["blaze_deg"], clicked["energy_ev"]) == (4.0, 0.9, 5000.0)
    assert (newest["d_nm"], newest["energy_ev"]) == (4.0, 5000.0)
    assert (first["d_nm"], first["energy_ev"]) == (3.5, 4895.0)
    assert energy_scan_scans(tmp_path, 9.0, 9.0, 1.0)["traces"] == []
