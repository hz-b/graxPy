# ruff: noqa: D100,D103
from __future__ import annotations

import json
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from grax.materials import available_material_symbols

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


def _study_form(**overrides: str) -> dict[str, str]:
    form = {
        "display_name": "Ru/B4C design",
        "d_min_nm": "2.0",
        "d_max_nm": "4.0",
        "d_points": "3",
        "blaze_min_deg": "0.6",
        "blaze_max_deg": "1.4",
        "blaze_points": "3",
        "energy_scan_points": "3",
        **_material_form(),
    }
    form.update(overrides)
    return form


def _install_fake_runners(monkeypatch: pytest.MonkeyPatch, *, survey_cells: int = 0) -> None:
    """Replace the two solver entry points with fakes that write the real artifacts.

    The fakes honour ``progress_callback`` and ``should_continue`` exactly like
    the library does, so the worker plumbing (progress bar, cooperative abort)
    is exercised without running a solve. ``survey_cells`` pads the survey loop
    so a test can abort it mid-flight.
    """

    from grax.multilayer_design import EnergyScanResult, MultilayerGratingDesigner, StageProgress

    def fake_survey(self, *, progress_callback=None, should_continue=None):  # noqa: ANN001
        config = self.config
        d_values = config.d_grid_nm()
        blaze_values = config.blaze_grid_deg()
        rows = []
        aborted = False
        for _ in range(max(1, survey_cells)):
            for d_spacing in d_values:
                for blaze in blaze_values:
                    if should_continue is not None and not should_continue():
                        aborted = True
                        break
                    efficiency = (
                        0.5
                        * np.exp(-(((d_spacing - 3.0) / 0.9) ** 2))
                        * np.exp(-(((blaze - 1.0) / 0.3) ** 2))
                    )
                    rows.append(
                        {
                            "d_nm": float(d_spacing),
                            "blaze_deg": float(blaze),
                            "bragg_estimate_deg": 1.357,
                            "incidence_angle_deg": 0.585,
                            "peak_efficiency": float(efficiency),
                            "precise_fwhm_deg": 0.03,
                            "edge_clipped": 0.0,
                        }
                    )
                    if progress_callback is not None:
                        progress_callback(
                            StageProgress(
                                "survey", len(rows), d_values.size * blaze_values.size, "cell"
                            )
                        )
                if aborted:
                    break
            if aborted:
                break

        config.survey_dir.mkdir(parents=True, exist_ok=True)
        config.plot_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(config.survey_dir / "survey.csv", index=False)
        for name in (
            "optimal_blaze_vs_d_spacing.png",
            "max_efficiency_vs_d_spacing.png",
            "efficiency_heatmap_d_vs_blaze.png",
        ):
            (config.plot_dir / name).write_bytes(b"png")
        return types.SimpleNamespace(
            aborted=aborted,
            plot_path=config.plot_dir / "optimal_blaze_vs_d_spacing.png",
            efficiency_plot_path=config.plot_dir / "max_efficiency_vs_d_spacing.png",
            heatmap_plot_path=config.plot_dir / "efficiency_heatmap_d_vs_blaze.png",
            combined_csv_path=config.survey_dir / "survey.csv",
        )

    def fake_energy_scan(self, pairs, *, progress_callback=None, should_continue=None):  # noqa: ANN001
        config = self.config
        designs = list(pairs)
        results = []
        for index, (d_spacing, blaze) in enumerate(designs):
            if should_continue is not None and not should_continue():
                break
            design_dir = config.energy_scan_dir / f"d{d_spacing:.3f}nm_blaze{blaze:.3f}deg"
            design_dir.mkdir(parents=True, exist_ok=True)
            frame = pd.DataFrame(
                {"energy_ev": [3000.0, 9000.0, 12000.0], "selected_efficiency": [0.2, 0.6, 0.4]}
            )
            summary_csv = design_dir / "multilayer_theta_search_summary.csv"
            frame.to_csv(summary_csv, index=False)
            config.plot_dir.mkdir(parents=True, exist_ok=True)
            titled = config.plot_dir / f"efficiency_vs_energy_d{d_spacing:.3f}_b{blaze:.3f}.png"
            titled.write_bytes(b"png")
            results.append(
                EnergyScanResult(
                    d_spacing_nm=d_spacing,
                    blaze_angle_deg=blaze,
                    output_dir=design_dir,
                    summary_csv_path=summary_csv,
                    all_orders_csv_path=design_dir / "multilayer_theta_search_all_orders.csv",
                    energy_efficiency_plot_path=design_dir / "energy.png",
                    titled_plot_path=titled,
                    results=frame,
                )
            )
            if progress_callback is not None:
                progress_callback(StageProgress("energy_scan", index + 1, len(designs), "design"))
        return results

    def fake_overlay(self, results):  # noqa: ANN001
        path = self.config.plot_dir / "efficiency_vs_energy_comparison.png"
        path.write_bytes(b"png")
        return path

    monkeypatch.setattr(MultilayerGratingDesigner, "run_survey", fake_survey)
    monkeypatch.setattr(MultilayerGratingDesigner, "run_energy_scan", fake_energy_scan)
    monkeypatch.setattr(MultilayerGratingDesigner, "plot_energy_scan_overlay", fake_overlay)


def _store(tmp_path: Path):  # noqa: ANN202
    from grax.web.multilayer_design_studies import MultilayerDesignStudyStore

    return MultilayerDesignStudyStore(tmp_path / "multilayer_designs")


def _wait_for_stage(store, study_id: str, stage: str, *, timeout: float = 5.0) -> str:  # noqa: ANN001
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = store.load(study_id)["stages"][stage]["status"]
        if status in {"completed", "failed", "aborted"}:
            return status
        time.sleep(0.02)
    return "timeout"


def _create_study(client, extra: dict[str, str] | None = None) -> str:  # noqa: ANN001
    response = client.post("/multilayer-design", data=_study_form(**(extra or {})))
    assert response.status_code == 302
    return response.headers["Location"].rsplit("/", 1)[-1]


def _client(tmp_path: Path):  # noqa: ANN202
    pytest.importorskip("flask")
    from grax.web.app import create_app

    return create_app(data_dir=tmp_path).test_client()


def test_homepage_and_nav_link_to_the_design_tab(tmp_path: Path) -> None:
    client = _client(tmp_path)

    assert b'href="/multilayer-design"' in client.get("/").data
    assert client.get("/multilayer-design").status_code == 200


def test_form_groups_fields_and_hides_scan_settings_behind_advanced(tmp_path: Path) -> None:
    client = _client(tmp_path)

    html = client.get("/multilayer-design/new").get_data(as_text=True)

    assert "<summary>Advanced</summary>" in html
    # The two nested ThetaSearchScanSettings blocks render as dotted field names.
    assert 'name="survey_scan_settings.rough_scan_points"' in html
    assert 'name="energy_scan_settings.rough_scan_points"' in html
    # The three dataclass sections are all present.
    assert "Shared - materials" in html
    assert "Survey - target &amp; grids" in html
    assert "Energy scan - energy grid" in html
    # Pre-survey "scan best only" opt-in.
    assert 'name="auto_energy_scan"' in html


def test_scan_settings_are_grouped_one_line_per_theta_search_pass() -> None:
    from grax.web.multilayer_design_studies import study_form_sections

    sections = dict(study_form_sections(advanced=True))

    for section in ("Survey - theta-search settings", "Energy scan - theta-search settings"):
        rows = sections[section]
        assert [label for label, _ in rows] == [
            "Rough pass",
            "Fine pass",
            "Final solve",
            "Peak & roughness",
        ]
        assert [len(specs) for _, specs in rows] == [5, 5, 3, 2]

    # Sections without sub-groups stay one ungrouped run, flowing as before.
    assert [label for label, _ in sections["Shared - runtime"]] == [""]


def test_scan_settings_rows_render_as_labelled_lines(tmp_path: Path) -> None:
    client = _client(tmp_path)

    html = client.get("/multilayer-design/new").get_data(as_text=True)

    assert '<p class="field-row-label">Rough pass</p>' in html
    assert '<p class="field-row-label">Final solve</p>' in html
    assert html.count('class="field-row"') == 8  # four rows x two scan blocks
    # Grouping is presentation only -- every input keeps its dotted name.
    assert html.count('name="survey_scan_settings.') == 15


def test_form_offers_the_whole_material_catalog(tmp_path: Path) -> None:
    client = _client(tmp_path)

    html = client.get("/multilayer-design/new").get_data(as_text=True)

    assert html.count('<option value="') >= len(available_material_symbols())
    assert '<option value="Au" data-density="19.282">' in html


def test_create_writes_a_manifest_with_the_nested_scan_settings(tmp_path: Path) -> None:
    client = _client(tmp_path)

    study_id = _create_study(client, {"survey_scan_settings.rough_scan_points": "15"})

    manifest = _store(tmp_path).load(study_id)
    assert manifest["config"]["d_points"] == 3
    assert manifest["config"]["survey_scan_settings"]["rough_scan_points"] == 15
    assert manifest["config"]["material_a"] == ["Ru", 12.1]
    assert manifest["auto_energy_scan"] == "none"
    assert set(manifest["stages"]) == {"survey", "energy_scan"}


def test_invalid_config_is_rejected(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post("/multilayer-design", data=_study_form(d_points="1"))

    assert response.status_code == 400


def test_survey_runs_and_the_detail_page_shows_the_three_plots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_runners(monkeypatch)
    client = _client(tmp_path)
    study_id = _create_study(client)

    assert client.post(f"/multilayer-design/{study_id}/stages/survey/run").status_code == 302
    assert _wait_for_stage(_store(tmp_path), study_id, "survey") == "completed"

    html = client.get(f"/multilayer-design/{study_id}").get_data(as_text=True)
    assert "optimal_blaze_vs_d_spacing.png" in html
    assert "max_efficiency_vs_d_spacing.png" in html
    assert "efficiency_heatmap_d_vs_blaze.png" in html
    assert "survey.csv" in html
    # All three step-2 choices are offered once the survey is done.
    assert 'value="best"' in html
    assert 'value="per_d"' in html
    assert "data-design-toggle" in html

    payload = client.get(f"/multilayer-design/{study_id}/stages/survey/status").get_json()
    assert {"state", "completed_points", "total_points", "plot_url", "can_abort"} <= set(payload)


def test_energy_scan_cannot_run_before_the_survey(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_runners(monkeypatch)
    client = _client(tmp_path)
    study_id = _create_study(client)

    response = client.post(
        f"/multilayer-design/{study_id}/stages/energy_scan/run", data={"mode": "best"}
    )

    assert response.status_code == 409


@pytest.mark.parametrize(
    ("mode", "form", "expected"),
    [
        ("best", {}, [[3.0, 1.0]]),
        ("per_d", {}, [[2.0, 1.0], [3.0, 1.0], [4.0, 1.0]]),
        ("manual", {"design": ["2.000,0.600", "4.000,1.400"]}, [[2.0, 0.6], [4.0, 1.4]]),
    ],
)
def test_each_selection_mode_scans_the_right_designs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
    form: dict[str, list[str]],
    expected: list[list[float]],
) -> None:
    _install_fake_runners(monkeypatch)
    client = _client(tmp_path)
    store = _store(tmp_path)
    study_id = _create_study(client)
    client.post(f"/multilayer-design/{study_id}/stages/survey/run")
    assert _wait_for_stage(store, study_id, "survey") == "completed"

    response = client.post(
        f"/multilayer-design/{study_id}/stages/energy_scan/run", data={"mode": mode, **form}
    )

    assert response.status_code == 302
    assert _wait_for_stage(store, study_id, "energy_scan") == "completed"
    assert store.load(study_id)["stages"]["energy_scan"]["designs"] == expected


def test_manual_designs_off_the_survey_grid_are_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_runners(monkeypatch)
    client = _client(tmp_path)
    study_id = _create_study(client)
    client.post(f"/multilayer-design/{study_id}/stages/survey/run")
    _wait_for_stage(_store(tmp_path), study_id, "survey")

    response = client.post(
        f"/multilayer-design/{study_id}/stages/energy_scan/run",
        data={"mode": "manual", "design": ["9.9,9.9"]},
    )

    assert response.status_code == 400


def test_multi_design_scan_renders_the_overlay_and_each_design(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_runners(monkeypatch)
    client = _client(tmp_path)
    store = _store(tmp_path)
    study_id = _create_study(client)
    client.post(f"/multilayer-design/{study_id}/stages/survey/run")
    _wait_for_stage(store, study_id, "survey")

    client.post(f"/multilayer-design/{study_id}/stages/energy_scan/run", data={"mode": "per_d"})
    assert _wait_for_stage(store, study_id, "energy_scan") == "completed"

    artifacts = store.load(study_id)["stages"]["energy_scan"]["artifacts"]
    assert artifacts["overlay_plot"].endswith("efficiency_vs_energy_comparison.png")
    assert len(artifacts["designs"]) == 3

    html = client.get(f"/multilayer-design/{study_id}").get_data(as_text=True)
    assert "efficiency_vs_energy_comparison.png" in html
    # One figure per design, each with its own summary-CSV link.
    assert html.count(">summary.csv</a>") == 3


def test_auto_energy_scan_chains_from_the_survey_without_a_click(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_runners(monkeypatch)
    client = _client(tmp_path)
    store = _store(tmp_path)
    study_id = _create_study(client, {"auto_energy_scan": "1"})
    assert store.load(study_id)["auto_energy_scan"] == "best"

    client.post(f"/multilayer-design/{study_id}/stages/survey/run")

    assert _wait_for_stage(store, study_id, "survey") == "completed"
    assert _wait_for_stage(store, study_id, "energy_scan") == "completed"
    assert store.load(study_id)["stages"]["energy_scan"]["designs"] == [[3.0, 1.0]]


def test_rerunning_the_survey_marks_the_energy_scan_stale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_runners(monkeypatch)
    client = _client(tmp_path)
    store = _store(tmp_path)
    study_id = _create_study(client)
    client.post(f"/multilayer-design/{study_id}/stages/survey/run")
    _wait_for_stage(store, study_id, "survey")
    client.post(f"/multilayer-design/{study_id}/stages/energy_scan/run", data={"mode": "best"})
    _wait_for_stage(store, study_id, "energy_scan")

    client.post(f"/multilayer-design/{study_id}/stages/survey/run")
    _wait_for_stage(store, study_id, "survey")

    assert store.load(study_id)["stages"]["energy_scan"]["status"] == "stale"


def test_reset_clears_one_stage_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_runners(monkeypatch)
    client = _client(tmp_path)
    store = _store(tmp_path)
    study_id = _create_study(client)
    client.post(f"/multilayer-design/{study_id}/stages/survey/run")
    _wait_for_stage(store, study_id, "survey")
    client.post(f"/multilayer-design/{study_id}/stages/energy_scan/run", data={"mode": "best"})
    _wait_for_stage(store, study_id, "energy_scan")

    client.post(f"/multilayer-design/{study_id}/stages/energy_scan/reset")

    manifest = store.load(study_id)
    assert manifest["stages"]["energy_scan"]["status"] == "not_run"
    assert manifest["stages"]["energy_scan"]["designs"] == []
    assert manifest["stages"]["survey"]["status"] == "completed"
    study_dir = store.study_dir(study_id)
    assert not (study_dir / "energy_scan").exists()
    assert (study_dir / "survey" / "survey.csv").is_file()


def test_abort_with_discard_removes_the_partial_survey(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_runners(monkeypatch, survey_cells=4000)
    client = _client(tmp_path)
    store = _store(tmp_path)
    study_id = _create_study(client)
    client.post(f"/multilayer-design/{study_id}/stages/survey/run")
    time.sleep(0.1)

    response = client.post(
        f"/multilayer-design/{study_id}/stages/survey/abort", data={"disposition": "discard"}
    )

    assert response.status_code == 302
    manifest = store.load(study_id)
    assert manifest["stages"]["survey"]["status"] == "not_run"
    assert not (store.study_dir(study_id) / "survey").exists()


def test_delete_removes_the_study_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_runners(monkeypatch)
    client = _client(tmp_path)
    study_id = _create_study(client)

    client.post(f"/multilayer-design/{study_id}/delete")

    assert not (tmp_path / "multilayer_designs" / study_id).exists()


def test_study_id_cannot_escape_the_data_directory(tmp_path: Path) -> None:
    client = _client(tmp_path)

    assert client.get("/multilayer-design/..%2f..%2fetc").status_code == 404


def test_design_options_are_embedded_for_the_manual_picker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_runners(monkeypatch)
    client = _client(tmp_path)
    study_id = _create_study(client)
    client.post(f"/multilayer-design/{study_id}/stages/survey/run")
    _wait_for_stage(_store(tmp_path), study_id, "survey")

    html = client.get(f"/multilayer-design/{study_id}").get_data(as_text=True)

    assert "data-design-options=" in html
    from grax.web.multilayer_design_studies import survey_design_options

    options = survey_design_options(_store(tmp_path).study_dir(study_id))
    assert options["d_values"] == [2.0, 3.0, 4.0]
    assert options["blaze_values"] == [0.6, 1.0, 1.4]
    assert options["best"] == [3.0, 1.0]
    assert json.dumps(options)  # the template embeds this verbatim


def test_energy_scan_progress_counts_solved_energies_not_designs(tmp_path: Path) -> None:
    # run_energy_scan reports once per design, so a single-design scan would sit
    # at 0 / 1 for hours. The monitor counts checkpoint lines instead.
    from grax.web.app import _energy_scan_checkpoint_progress

    checkpoints = tmp_path / "energy_scan" / "d3.000nm_blaze0.800deg" / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "results.jsonl").write_text(
        '{"energy_ev": 1000}\n{"energy_ev": 1010}\n\n', encoding="utf-8"
    )

    completed, total = _energy_scan_checkpoint_progress(
        study_dir=tmp_path, designs=[[3.0, 0.8]], energy_points=200
    )

    assert (completed, total) == (2, 200)


def test_energy_scan_progress_is_zero_before_any_checkpoint(tmp_path: Path) -> None:
    from grax.web.app import _energy_scan_checkpoint_progress

    completed, total = _energy_scan_checkpoint_progress(
        study_dir=tmp_path, designs=[[3.0, 0.8], [4.0, 1.2]], energy_points=5
    )

    assert (completed, total) == (0, 10)
