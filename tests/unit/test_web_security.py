# ruff: noqa: D100,D103

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("flask")

from grax.web import app as web_app_module  # noqa: E402
from grax.web.app import create_app  # noqa: E402

SETTINGS = {"decimal_separator": "comma", "standard_polarization": "s"}


@pytest.fixture
def client(tmp_path: Path):
    return create_app(data_dir=tmp_path).test_client()


def test_cross_origin_posts_are_rejected(client) -> None:
    response = client.post("/settings", json=SETTINGS, headers={"Origin": "http://evil.example"})

    assert response.status_code == 403


def test_cross_site_fetch_metadata_is_rejected_without_origin(client) -> None:
    response = client.post("/settings", json=SETTINGS, headers={"Sec-Fetch-Site": "cross-site"})

    assert response.status_code == 403


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Origin": "http://localhost"},
        {"Origin": "http://localhost", "Sec-Fetch-Site": "same-origin"},
        {"Origin": "null", "Sec-Fetch-Site": "none"},
    ],
)
def test_same_origin_and_scripted_posts_still_work(client, headers) -> None:
    response = client.post("/settings", json=SETTINGS, headers=headers)

    assert response.status_code == 200


def test_safe_methods_are_not_checked(client) -> None:
    assert client.get("/", headers={"Origin": "http://evil.example"}).status_code == 200


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/plots/%2E%2E/delete"),
        ("get", "/plots/%2E%2E/delete"),
        ("post", "/runs/%2E%2E/restart"),
        ("post", "/runs/%2E%2E/abort"),
        ("get", "/runs/%2E%2E/status"),
        ("post", "/gratings/%2E%2E/delete"),
        ("post", "/multilayer-design/%2E%2E/delete"),
        ("get", "/plane-mirror-runs/%2E%2E/status"),
        ("post", "/plane-mirrors/demo/scans/%2E%2E/delete"),
    ],
)
def test_path_like_ids_are_not_found_and_touch_no_files(tmp_path: Path, method: str, path: str) -> None:
    client = create_app(data_dir=tmp_path / "workspace").test_client()
    workspace = tmp_path / "workspace"
    for folder in ("plots", "runs", "saved_gratings", "multilayer_designs", "saved_plane_mirrors"):
        (workspace / folder).mkdir(parents=True)
    # A manifest one level above each store would be picked up by an unchecked "..".
    sentinel = {"id": "x", "run_input": {}, "workflow": "energy_sweep", "mirror_id": "m", "status": "completed"}
    (workspace / "manifest.json").write_text(json.dumps(sentinel), encoding="utf-8")
    (workspace / "keep.txt").write_text("keep", encoding="utf-8")
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))

    response = getattr(client, method)(path, data={"disposition": "delete"})

    assert response.status_code == 404
    assert sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")) == before


def _run_main(monkeypatch: pytest.MonkeyPatch, *argv: str) -> dict:
    calls: dict = {}
    fake_app = SimpleNamespace(run=lambda **kwargs: calls.update(kwargs))
    monkeypatch.setattr(web_app_module, "create_app", lambda **kwargs: calls.update(app_kwargs=kwargs) or fake_app)
    monkeypatch.setattr(
        web_app_module,
        "_maybe_open_browser",
        lambda host, port, **kwargs: calls.update(browser=(host, port), browser_options=kwargs),
    )
    monkeypatch.setattr(sys, "argv", ["grax-web", *argv])
    web_app_module.main()
    return calls


def test_server_runs_without_debugger_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _run_main(monkeypatch)

    assert calls["debug"] is False
    assert calls["host"] == "127.0.0.1"
    assert calls["browser"] == ("127.0.0.1", 5050)


def test_data_dir_option_selects_the_database_folder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _run_main(monkeypatch, "--data-dir", str(tmp_path))

    assert calls["app_kwargs"] == {"data_dir": tmp_path}


def test_debug_flag_enables_debugger_on_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run_main(monkeypatch, "--debug")["debug"] is True


def test_debug_flag_is_refused_on_network_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, "--debug", "--host", "0.0.0.0")


def test_browser_opens_once_without_the_reloader(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    monkeypatch.delenv("WERKZEUG_RUN_MAIN", raising=False)
    monkeypatch.delenv("FLASK_RUN_FROM_CLI", raising=False)
    monkeypatch.setattr(web_app_module.webbrowser, "open", opened.append)
    monkeypatch.setattr(
        web_app_module.threading,
        "Timer",
        lambda delay, callback: SimpleNamespace(daemon=False, start=callback),
    )

    web_app_module._maybe_open_browser("0.0.0.0", 5050)

    assert opened == ["http://127.0.0.1:5050"]


def test_browser_opens_only_in_the_reloader_child(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr(web_app_module.webbrowser, "open", opened.append)
    monkeypatch.setattr(
        web_app_module.threading,
        "Timer",
        lambda delay, callback: SimpleNamespace(daemon=False, start=callback),
    )

    monkeypatch.delenv("WERKZEUG_RUN_MAIN", raising=False)
    web_app_module._maybe_open_browser("127.0.0.1", 5050, reloader=True)
    assert opened == []

    monkeypatch.setenv("WERKZEUG_RUN_MAIN", "true")
    web_app_module._maybe_open_browser("127.0.0.1", 5050, reloader=True)
    assert opened == ["http://127.0.0.1:5050"]
