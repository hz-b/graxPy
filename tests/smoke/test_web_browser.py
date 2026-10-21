"""Browser checks for the Flask website, including one real solver run."""

from __future__ import annotations

import csv
import math
import os
import shutil
import threading
import time
from pathlib import Path

import pytest
from playwright.sync_api import Browser, Page, expect, sync_playwright
from werkzeug.serving import make_server

from grax.web.app import create_app


@pytest.fixture
def website(tmp_path: Path):
    """Run an isolated Flask instance on an ephemeral local port."""
    app = create_app(data_dir=tmp_path / "web-data")
    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", tmp_path / "web-data"
    finally:
        server.shutdown()
        thread.join(timeout=5)


@pytest.fixture
def chromium() -> Browser:
    """Start a headless Chromium instance for this test."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            yield browser
        finally:
            browser.close()


@pytest.fixture
def browser_page(website, chromium: Browser, tmp_path: Path, request) -> Page:
    """Keep a trace and run files when a browser check fails."""
    base_url, data_dir = website
    context = chromium.new_context(accept_downloads=True)
    context.tracing.start(screenshots=True, snapshots=True, sources=True)
    page = context.new_page()
    try:
        page.goto(base_url)
        yield page
    except Exception:
        artifact_root = os.environ.get(
            "GRAX_TEST_ARTIFACTS", str(tmp_path / "browser-failure-artifacts")
        )
        artifacts = Path(artifact_root)
        artifacts.mkdir(exist_ok=True)
        page.screenshot(path=artifacts / "failure.png", full_page=True)
        context.tracing.stop(path=artifacts / "trace.zip")
        run_dir = data_dir / "runs"
        if run_dir.exists():
            shutil.copytree(run_dir, artifacts / "runs", dirs_exist_ok=True)
        request.node.user_properties.append(("browser_failure_artifacts", str(artifacts)))
        raise
    else:
        context.tracing.stop()
    finally:
        context.close()


@pytest.mark.browser
def test_navigation_and_grating_form_use_javascript(browser_page: Page) -> None:
    """Keep primary navigation and conditional grating fields interactive."""
    page = browser_page
    page.locator("details.nav-dropdown").nth(0).locator("summary").click()
    page.get_by_role("link", name="New grating").first.click()
    expect(page.get_by_role("heading", name="New grating")).to_be_visible()
    expect(page.locator("[data-grating-section='blazed']")).to_be_hidden()

    page.locator("select[name='grating_type']").select_option("blazed")
    expect(page.locator("[data-grating-section='blazed']")).to_be_visible()
    expect(page.locator("[data-grating-section='laminar']")).to_be_hidden()

    page.locator("details.nav-dropdown").nth(0).locator("summary").click()
    page.locator("details.nav-dropdown").nth(1).locator("summary").click()
    expect(page.locator("details.nav-dropdown").nth(0)).not_to_have_attribute("open", "")
    expect(page.locator("details.nav-dropdown").nth(1)).to_have_attribute("open", "")


@pytest.mark.browser
@pytest.mark.smoke
def test_browser_runs_real_two_point_energy_sweep_and_downloads_csv(
    website, browser_page: Page
) -> None:
    """Run a real two-point sweep in Chromium and validate its downloaded data."""
    base_url, data_dir = website
    page = browser_page
    page.goto(f"{base_url}/gratings/new")

    page.locator("input[name='name']").fill("CI energy scan")
    page.locator("select[name='grating_type']").select_option("blazed")
    page.locator("input[name='period_lpermm']").fill("600")
    page.locator("input[name='blaze_angle_deg']").fill("0.75")
    page.locator("input[name='x_resolution_nm']").fill("4")
    page.locator("input[name='z_resolution_nm']").fill("4")
    page.locator("select[name='stack_type']").select_option("single_layer")
    page.locator("input[name='substrate_material']").fill("Si")
    page.locator("input[name='layer_material']").fill("Au")
    page.locator("input[name='layer_thickness_nm']").fill("30")
    page.get_by_role("button", name="Save grating").click()
    expect(page.get_by_role("heading", name="CI energy scan")).to_be_visible(timeout=20_000)

    page.locator("select[name='workflow']").select_option("energy_sweep")
    page.locator("input[name='energy_start_ev']").fill("800")
    page.locator("input[name='energy_stop_ev']").fill("810")
    page.locator("input[name='energy_points']").fill("2")
    page.locator("input[name='grazing_angle_deg']").fill("1.5")
    page.locator("input[name='fourier_orders']").fill("3")
    page.locator("input[name='run_x_resolution_nm']").fill("4")
    page.locator("input[name='run_z_resolution_nm']").fill("4")
    page.locator("select[name='max_workers_mode']").select_option("manual")
    page.locator("input[name='max_workers']").fill("1")
    page.get_by_role("button", name="Run sweep").click()

    run_id = page.url.rstrip("/").rsplit("/", 1)[-1]
    deadline = time.monotonic() + 90
    status = {}
    while time.monotonic() < deadline:
        status_response = page.request.get(f"{base_url}/runs/{run_id}/status")
        status = status_response.json()
        if status["state"] in {"completed", "failed", "error", "aborted"}:
            break
        page.wait_for_timeout(250)
    assert status["state"] == "completed", status
    expect(page.locator("[data-run-state]")).to_have_text("completed")
    assert status["completed_points"] == 2, status
    assert status["total_points"] == 2, status

    run_dir = data_dir / "runs" / run_id
    plot_path = run_dir / "selected_efficiency.png"
    assert plot_path.is_file(), f"Run status: {status}; artifacts: {list(run_dir.glob('*'))}"
    assert plot_path.stat().st_size > 1_000, f"Run status: {status}; plot is empty"

    with page.expect_download() as download_info:
        page.get_by_role("link", name="Download CSV").click()
    download = download_info.value
    csv_path = run_dir / download.suggested_filename
    download.save_as(csv_path)
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["case_id"] for row in rows} == {
        "energy_sweep-00000000",
        "energy_sweep-00000001",
    }, f"Run status: {status}; downloaded rows: {rows}"
    for row in rows:
        efficiency = float(row["efficiency"])
        assert math.isfinite(efficiency) and efficiency >= 0, row
