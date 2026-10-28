# ruff: noqa: D100,D103

from __future__ import annotations

import json
from pathlib import Path

from grax.web.app import create_app

INVENTORY = Path(__file__).parent / "fixtures" / "web_route_inventory.json"


def _route_inventory(app) -> list[list]:
    return sorted(
        [rule.rule, sorted(rule.methods - {"HEAD", "OPTIONS"}), rule.endpoint]
        for rule in app.url_map.iter_rules()
    )


def test_web_routes_keep_their_urls_methods_and_endpoint_names(tmp_path):
    """Templates and bookmarks rely on these; restructuring must not change them."""
    expected = json.loads(INVENTORY.read_text(encoding="utf-8"))

    assert _route_inventory(create_app(data_dir=tmp_path)) == expected
