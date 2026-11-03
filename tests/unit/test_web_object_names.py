"""Renaming a saved object reaches its runs; Fourier hints and the grating guide render."""

# ruff: noqa: D103
from __future__ import annotations

from pathlib import Path

import pytest

from grax.web.multilayer_design_studies import MultilayerDesignStudyStore, MultilayerGratingStore
from grax.web.object_names import _rename_in_title, rename_grating_runs
from grax.web.runs import RunStore

from .test_web_multilayer_design import _client, _create_study, _study_form
from .test_web_pgm import (
    MIRROR_FORM,
    _merged,
    pgm_client,  # noqa: F401
)

pytestmark = pytest.mark.unit


def _save_run(tmp_path: Path, folder: str = "runs", **fields: object) -> dict:
    return RunStore(tmp_path / folder).save({"status": "completed", **fields})


def _load_run(tmp_path: Path, run_id: str, folder: str = "runs") -> dict:
    return RunStore(tmp_path / folder).load(run_id)


def test_title_replacement_matches_whole_names_only() -> None:
    assert _rename_in_title("Pt · energy_sweep", {"Pt"}, "Ir") == "Ir · energy_sweep"
    assert _rename_in_title("Ptolemy Pt test", {"Pt"}, "Ir") == "Ptolemy Ir test"
    assert _rename_in_title("(a+b) run", {"(a+b)"}, "c") == "c run"
    assert _rename_in_title(r"old run", {"old"}, r"new\1") == r"new\1 run"
    assert _rename_in_title("My own title", {"", "old"}, "new") == "My own title"


def test_renaming_a_grating_updates_its_runs_and_leaves_others_alone(pgm_client, tmp_path) -> None:  # noqa: F811
    _, client, grating, _ = pgm_client
    old = grating["name"]
    _save_run(
        tmp_path,
        id="auto",
        grating_id=grating["id"],
        grating_name=old,
        grating_spec={"name": old},
        workflow="energy_sweep",
        display_name=f"{old} · energy_sweep",
    )
    _save_run(
        tmp_path,
        id="custom",
        grating_id=grating["id"],
        grating_name=old,
        grating_spec={"name": old},
        display_name=f"Best {old} result",
    )
    _save_run(
        tmp_path,
        id="other",
        grating_id="another",
        grating_name=old,
        grating_spec={"name": old},
        display_name=f"{old} · energy_sweep",
    )
    _save_run(
        tmp_path,
        "pgm_runs",
        id="pgm",
        grating_id=grating["id"],
        grating_name=old,
        grating_snapshot={"name": old},
        display_name="PGM working range",
    )

    assert (
        client.post(f"/gratings/{grating['id']}/rename", data={"name": "Renamed"}).status_code
        == 302
    )

    auto = _load_run(tmp_path, "auto")
    assert (auto["grating_name"], auto["grating_spec"]["name"]) == ("Renamed", "Renamed")
    assert auto["display_name"] == "Renamed · energy_sweep"
    assert _load_run(tmp_path, "custom")["display_name"] == "Best Renamed result"
    other = _load_run(tmp_path, "other")
    assert (other["grating_name"], other["display_name"]) == (old, f"{old} · energy_sweep")
    linked_pgm = _load_run(tmp_path, "pgm", "pgm_runs")
    assert (linked_pgm["grating_name"], linked_pgm["grating_snapshot"]["name"]) == (
        "Renamed",
        "Renamed",
    )
    assert linked_pgm["display_name"] == "PGM working range"

    page = client.get("/runs/gratings").get_data(as_text=True)
    assert "Renamed · energy_sweep" in page
    assert f">{old} · energy_sweep<" in page  # Only the unrelated run keeps the old name.


def test_editing_a_grating_name_reaches_its_runs(pgm_client, tmp_path) -> None:  # noqa: F811
    from .test_web_pgm import GRATING_FIELDS

    _, client, grating, _ = pgm_client
    _save_run(
        tmp_path,
        id="run",
        grating_id=grating["id"],
        grating_name=grating["name"],
        display_name=f"{grating['name']} · angle_sweep",
    )

    response = client.post(f"/gratings/{grating['id']}", data={**GRATING_FIELDS, "name": "Edited"})

    assert response.status_code == 302
    run = _load_run(tmp_path, "run")
    assert (run["grating_name"], run["display_name"]) == ("Edited", "Edited · angle_sweep")


def test_stale_runs_correct_themselves_on_the_next_rename(tmp_path: Path) -> None:
    _save_run(
        tmp_path,
        id="stale",
        grating_id="g1",
        grating_name="Oldest",
        display_name="Oldest · energy_sweep",
    )

    rename_grating_runs(tmp_path, "g1", "Middle", "Newest")

    run = _load_run(tmp_path, "stale")
    assert (run["grating_name"], run["display_name"]) == ("Newest", "Newest · energy_sweep")


def test_an_unchanged_name_writes_nothing(tmp_path: Path) -> None:
    _save_run(tmp_path, id="stale", grating_id="g1", grating_name="Old", display_name="Old · run")

    rename_grating_runs(tmp_path, "g1", "Same", "Same")

    assert _load_run(tmp_path, "stale")["grating_name"] == "Old"


def test_renaming_a_mirror_updates_its_runs(pgm_client, tmp_path) -> None:  # noqa: F811
    _, client, _, mirror = pgm_client
    old = mirror["name"]
    _save_run(
        tmp_path,
        id="scan",
        mirror_id=mirror["id"],
        mirror_name=old,
        mirror_snapshot={"name": old},
        display_name=f"{old} · energy scan",
    )
    _save_run(
        tmp_path,
        "pgm_runs",
        id="pgm",
        mirror_id=mirror["id"],
        mirror_name=old,
        mirror_snapshot={"name": old},
        display_name="PGM working range",
    )

    response = client.post(
        f"/plane-mirrors/{mirror['id']}", data={**MIRROR_FORM, "mirror_name": "New mirror"}
    )

    assert response.status_code == 302
    scan = _load_run(tmp_path, "scan")
    assert (scan["mirror_name"], scan["mirror_snapshot"]["name"]) == ("New mirror", "New mirror")
    assert scan["display_name"] == "New mirror · energy scan"
    assert _load_run(tmp_path, "pgm", "pgm_runs")["mirror_name"] == "New mirror"
    assert "New mirror · energy scan" in client.get("/plane-mirror-runs").get_data(as_text=True)


def test_renaming_a_saved_pgm_updates_its_runs(pgm_client, tmp_path) -> None:  # noqa: F811
    _, client, _, _ = pgm_client
    saved = client.post("/pgm/saved", data=_merged(pgm_name="My PGM")).get_json()
    _save_run(
        tmp_path,
        "pgm_runs",
        id="run",
        pgm_id=saved["id"],
        pgm_name="My PGM",
        display_name="My PGM at 600",
    )
    _save_run(
        tmp_path,
        "pgm_runs",
        id="other",
        pgm_id="sx700",
        pgm_name="My PGM",
        display_name="My PGM at 1200",
    )

    assert (
        client.post(f"/pgm/saved/{saved['id']}/rename", data={"name": "Beamline PGM"}).status_code
        == 302
    )

    run = _load_run(tmp_path, "run", "pgm_runs")
    assert (run["pgm_name"], run["display_name"]) == ("Beamline PGM", "Beamline PGM at 600")
    assert _load_run(tmp_path, "other", "pgm_runs")["display_name"] == "My PGM at 1200"


def test_renaming_a_multilayer_grating_updates_study_titles(tmp_path: Path) -> None:
    client = _client(tmp_path)
    studies = MultilayerDesignStudyStore(tmp_path / "multilayer_designs")
    gratings = MultilayerGratingStore(tmp_path / "multilayer_gratings")
    study_id = _create_study(client)
    grating = gratings.load(studies.load(study_id)["grating_id"])
    old = grating["display_name"]
    studies.update(study_id, lambda saved: saved.update(display_name=f"{old} survey"))

    response = client.post(
        f"/multilayer-gratings/{grating['id']}/edit", data=_study_form(display_name="Renamed ML")
    )

    assert response.status_code == 302
    assert gratings.load(grating["id"])["display_name"] == "Renamed ML"
    assert studies.load(study_id)["display_name"] == "Renamed ML survey"


def test_every_fourier_orders_field_has_the_hint(pgm_client, tmp_path) -> None:  # noqa: F811
    _, client, grating, _ = pgm_client
    hint = "at least as many Fourier orders as the diffraction order"

    detail = client.get(f"/gratings/{grating['id']}").get_data(as_text=True)
    assert detail.count(hint) == 5
    for key in (
        "fourier_orders",
        "study_x_fixed_fourier",
        "study_z_fixed_fourier",
        "custom_default_fourier_orders",
    ):
        assert f'id="fourier-help-{key}"' in detail

    assert client.get("/pgm").get_data(as_text=True).count(hint) == 1

    multilayer = client.get("/multilayer-design/new").get_data(as_text=True)
    assert multilayer.count(hint) == 6
    assert 'id="fourier-help-survey_scan_settings-rough_fourier_orders"' in multilayer


def test_grating_guide_is_linked_and_menu_says_new_pgm(pgm_client) -> None:  # noqa: F811
    _, client, _, _ = pgm_client

    home = client.get("/").get_data(as_text=True)
    assert 'href="/docs/gratings-how-it-works"' in home
    assert ">New PGM</a>" in home and "New working range" not in home
    assert 'href="/docs/gratings-how-it-works"' in client.get("/docs/gratings").get_data(
        as_text=True
    )

    guide = client.get("/docs/gratings-how-it-works")
    assert guide.status_code == 200
    text = guide.get_data(as_text=True)
    assert "Parameter study" in text and "smallest feature" in text
