# ruff: noqa: D100,D103

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import pytest

from tests.unit.test_saved_plot_loops import _write_sweep
from tests.unit.test_web_multilayer_design import _study_form
from tests.unit.test_web_plane_mirror import FORM as PLANE_MIRROR_FORM

pytest.importorskip("flask")

from grax.web.app import create_app  # noqa: E402

GRATING_FORM = {
    "name": "Sweep laminar",
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
    "roughness_kind": "none",
}

# Plotly is inlined into pages, so links inside it are skipped by restricting to href/src/action.
LINK_PATTERN = re.compile(r'(?:href|src)="([^"#]+)"')


@pytest.fixture
def seeded_client(tmp_path: Path):
    app = create_app(data_dir=tmp_path)
    client = app.test_client()
    assert client.post("/gratings", data=GRATING_FORM).status_code == 302
    assert client.post("/plane-mirrors", data={**PLANE_MIRROR_FORM, "mirror_name": "Sweep mirror"}).status_code == 302
    assert client.post("/multilayer-design", data=_study_form()).status_code == 302
    _write_sweep(tmp_path)
    return client


def _crawl(client, start: str = "/") -> dict[str, int]:
    seen: dict[str, int] = {}
    queue = [start]
    while queue:
        url = queue.pop()
        if url in seen:
            continue
        response = client.get(url)
        seen[url] = response.status_code
        if response.status_code != 200 or not response.mimetype.startswith("text/html"):
            continue
        for link in LINK_PATTERN.findall(response.get_data(as_text=True)):
            target = urljoin(url, link.replace("&amp;", "&"))
            parts = urlsplit(target)
            if parts.scheme or parts.netloc or parts.path.startswith("/static/"):
                continue
            queue.append(parts.path + (f"?{parts.query}" if parts.query else ""))
    return seen


def test_every_linked_page_renders_without_server_errors(seeded_client):
    statuses = _crawl(seeded_client)

    failures = {url: status for url, status in statuses.items() if status >= 500}
    assert failures == {}
    # The sweep must reach every section, not just the index.
    for prefix in ("/gratings/", "/plane-mirrors/", "/multilayer-design/", "/runs/", "/plots", "/docs"):
        assert any(url.startswith(prefix) and status == 200 for url, status in statuses.items()), prefix
