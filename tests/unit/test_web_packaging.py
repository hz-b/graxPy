# ruff: noqa: D100,D103

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_package_data_covers_every_website_file():
    """Files a wheel leaves out work from the repository but are missing once installed."""
    with (ROOT / "pyproject.toml").open("rb") as stream:
        patterns = tomllib.load(stream)["tool"]["setuptools"]["package-data"]["grax.web"]
    web = ROOT / "src" / "grax" / "web"
    packaged = {path for pattern in patterns for path in web.glob(pattern)}
    needed = {
        path
        for folder in ("static", "templates")
        for path in (web / folder).rglob("*")
        if path.is_file() and path.name != ".DS_Store"
    }

    assert sorted(str(path.relative_to(web)) for path in needed - packaged) == []
