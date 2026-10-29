# ruff: noqa: D100,D103
from __future__ import annotations

from pathlib import Path

import pytest

from grax.web.multilayer_design_studies import MultilayerDesignStudyStore, MultilayerGratingStore

from .test_web_multilayer_design import _client, _create_study, _study_form

pytestmark = pytest.mark.unit


def _stores(tmp_path: Path):  # noqa: ANN202
    return (
        MultilayerDesignStudyStore(tmp_path / "multilayer_designs"),
        MultilayerGratingStore(tmp_path / "multilayer_gratings"),
    )


def test_identical_studies_share_a_grating_and_a_different_one_gets_its_own(tmp_path: Path) -> None:
    client = _client(tmp_path)
    studies, gratings = _stores(tmp_path)
    first = _create_study(client)
    second = _create_study(client, {"display_name": "Second"})
    other = _create_study(client, {"grating_density_lpermm": "1800"})

    ids = {key: studies.load(key)["grating_id"] for key in (first, second, other)}

    assert ids[first] == ids[second] != ids[other]
    assert len(gratings.list()) == 2


def test_manage_page_shows_preview_and_attached_studies(tmp_path: Path) -> None:
    client = _client(tmp_path)
    first = _create_study(client)
    _create_study(client, {"display_name": "Second"})

    html = client.get("/multilayer-gratings").get_data(as_text=True)

    assert "data-geometry-view" in html
    assert "2 studies" in html and "Second" in html and f"/multilayer-design/{first}" in html
    assert "Manage multilayer gratings" in client.get("/").get_data(as_text=True)


def test_grating_opens_its_first_study_and_the_study_page_lists_siblings(tmp_path: Path) -> None:
    client = _client(tmp_path)
    studies, _ = _stores(tmp_path)
    older = _create_study(client)
    studies.update(older, lambda m: m.update(created_at="2020-01-01T00:00:00"))
    newest = _create_study(client, {"display_name": "Newest"})
    grating_id = studies.load(newest)["grating_id"]

    response = client.get(f"/multilayer-gratings/{grating_id}")
    page = client.get(f"/multilayer-design/{newest}").get_data(as_text=True)

    assert response.status_code == 302 and response.headers["Location"].endswith(newest)
    assert page.count("<option value=\"/multilayer-design/") == 2
    assert f"grating={grating_id}" in page


def test_new_study_for_a_grating_inherits_its_geometry(tmp_path: Path) -> None:
    client = _client(tmp_path)
    studies, _ = _stores(tmp_path)
    grating_id = studies.load(_create_study(client, {"grating_density_lpermm": "1800"}))["grating_id"]

    form = client.get("/multilayer-design/new", query_string={"grating": grating_id}).get_data(as_text=True)
    created = client.post(
        "/multilayer-design",
        data={"grating_id": grating_id, "display_name": "Third", "d_min_nm": "2.5"},
    )
    study = studies.load(created.headers["Location"].rsplit("/", 1)[-1])

    assert 'name="grating_density_lpermm"' not in form and f'value="{grating_id}"' in form
    assert study["grating_id"] == grating_id
    assert study["config"]["grating_density_lpermm"] == 1800.0 and study["config"]["d_min_nm"] == 2.5


def test_editing_a_grating_updates_its_studies_and_marks_results_stale(tmp_path: Path) -> None:
    client = _client(tmp_path)
    studies, gratings = _stores(tmp_path)
    study_id = _create_study(client)
    grating_id = studies.load(study_id)["grating_id"]
    studies.update(study_id, lambda m: m["stages"]["survey"].update(status="completed"))

    response = client.post(
        f"/multilayer-gratings/{grating_id}/edit",
        data={**_study_form(), "display_name": "Renamed", "grating_density_lpermm": "1500"},
    )

    manifest = studies.load(study_id)
    assert response.status_code == 302
    assert gratings.load(grating_id)["display_name"] == "Renamed"
    assert manifest["config"]["grating_density_lpermm"] == 1500.0
    assert manifest["stages"]["survey"]["status"] == "stale"


def test_deleting_a_grating_deletes_its_studies(tmp_path: Path) -> None:
    client = _client(tmp_path)
    studies, gratings = _stores(tmp_path)
    study_id = _create_study(client)
    grating_id = studies.load(study_id)["grating_id"]

    client.post(f"/multilayer-gratings/{grating_id}/delete")

    assert gratings.list() == [] and studies.list() == []
    assert client.get("/multilayer-gratings/nope").status_code == 404
