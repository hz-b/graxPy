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
from tests.unit.test_saved_plot_loops import _write_sweep


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
def test_afm_upload_status_panel_reports_errors_and_copies_output(browser_page: Page) -> None:
    """Keep AFM progress visible beside compact inputs and make long errors usable."""
    page = browser_page
    page.goto(f"{page.url.rstrip('/')}/gratings/new")
    page.locator("select[name='grating_type']").select_option("afm")
    expect(page.locator("[data-afm-result]")).to_contain_text("Waiting for file")
    fields = page.locator(".afm-upload-fields").bounding_box()
    status = page.locator(".afm-status-panel").bounding_box()
    assert fields and status and status["x"] > fields["x"] + fields["width"]

    bad_file = Path(__file__).resolve().parents[2] / "examples/grating/data/bad_afm_upload_test.txt"
    page.locator("input[name='afm_file']").set_input_files(str(bad_file))
    expect(page.locator("[data-afm-result]")).to_contain_text("numeric", timeout=10_000)
    expect(page.locator("[data-afm-retry]")).to_be_visible()
    page.context.grant_permissions(
        ["clipboard-read", "clipboard-write"], origin=page.evaluate("location.origin")
    )
    page.locator("[data-afm-copy-output]").click()
    expect(page.locator("[data-afm-copy-output]")).to_have_text("Copied")
    copied_error = page.evaluate("navigator.clipboard.readText()")
    assert "AFM file must contain at least three numeric rows." in copied_error
    assert "Full traceback:" in copied_error
    assert "Traceback (most recent call last)" in copied_error
    page.locator("[data-afm-result]").evaluate(
        "node => { node.textContent = 'Failure details\\n' + 'x'.repeat(12000); }"
    )
    assert page.locator("[data-afm-result]").evaluate("node => node.scrollHeight > node.clientHeight")
    assert page.locator("[data-afm-result]").evaluate("node => getComputedStyle(node).overflowY") == "auto"
    page.locator("[data-afm-copy-output]").click()
    expect(page.locator("[data-afm-copy-output]")).to_have_text("Copied")
    copied = page.evaluate("navigator.clipboard.readText()")
    assert copied.startswith("Failure details") and len(copied) > 10_000


@pytest.mark.browser
def test_constant_deviation_angle_is_shown_only_for_that_mount(browser_page: Page) -> None:
    """Show a static 2θ marker on the nominal grating schematic only when selected."""
    page = browser_page
    page.goto(f"{page.url.rstrip('/')}/gratings/new")
    page.locator("input[name='name']").fill("Deviation schematic check")
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
    expect(page.locator("[data-run-form]")).to_be_visible(timeout=20_000)
    angle_label = page.locator(".geo-deviation-text")
    expect(angle_label).to_have_count(0)
    page.locator("select[name='workflow']").select_option("constant_deviation")
    expect(angle_label).to_have_text("2θ")
    angle = page.locator("input[name='total_deviation_deg']")
    angle.fill("174.0")
    assert angle.evaluate("node => node.validity.valid")
    page.locator("select[name='workflow']").select_option("energy_sweep")
    expect(angle_label).to_have_count(0)


@pytest.mark.browser
def test_decimal_preference_persists_after_reload(browser_page: Page) -> None:
    """Keep the chosen decimal separator in browser storage across page loads."""
    page = browser_page
    picker = page.locator("select[data-decimal-separator]")
    expect(picker).to_have_value("dot")
    picker.select_option("comma")
    page.reload()
    expect(page.locator("select[data-decimal-separator]")).to_have_value("comma")
    assert page.evaluate("localStorage.getItem('grax:decimal-separator')") == "comma"
    page.goto(f"{page.url.rstrip('/')}/gratings/new")
    page.locator("select[name='grating_type']").select_option("blazed")
    angle = page.locator("input[name='blaze_angle_deg']")
    angle.fill("")
    angle.press_sequentially("4,5")
    expect(angle).to_have_value("4.5")


@pytest.mark.browser
def test_decimal_comma_in_coupled_grid(browser_page: Page) -> None:
    """Accept the reported 4,5 value and keep the grid preview usable."""
    page = browser_page
    page.locator("select[data-decimal-separator]").select_option("comma")
    page.goto(f"{page.url.rstrip('/')}/gratings/new")
    page.locator("input[name='name']").fill("Comma grid check")
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
    expect(page.locator("[data-run-form]")).to_be_visible(timeout=20_000)
    page.locator("select[name='workflow']").select_option("energy_sweep")
    for name, value in (("energy_start_ev", "103.37"), ("energy_stop_ev", "1234.56"), ("grazing_angle_deg", "1.505")):
        field = page.locator(f"input[name='{name}']:visible").first
        field.fill(value)
        assert field.evaluate("node => node.validity.valid"), name
    stop = page.locator("input[name='energy_stop_ev']:visible").first
    stop.fill("100")
    assert stop.get_attribute("step") == "0.1"
    stop.press("ArrowUp")
    assert stop.input_value() == "100.1"
    page.locator("select[name='workflow']").select_option("custom_grid")
    builder = page.locator("[data-workflow-fields=custom_grid][data-custom-grid-builder]")
    builder.locator("[data-custom-add-axis]").click()
    axis = builder.locator("[data-custom-axis]").first
    axis.locator(".custom-grid-parameter").select_option("blaze_angle_deg")
    axis.locator("[data-axis-mode]").select_option("range")
    step = axis.locator("[data-linked-step]")
    step.fill("0.2")
    assert step.evaluate("node => node.validity.valid")
    axis.locator("[data-axis-mode]").select_option("list")
    axis.locator("textarea").fill("0,2; 0,4")
    builder.locator("[data-custom-add-group]").click()
    group = builder.locator("[data-custom-group]").first
    group.locator(".custom-grid-parameter").select_option("anti_blaze_angle_deg")
    group.locator("tbody input").first.fill("4,5")
    group.locator("tbody input").nth(1).fill("7")
    count_text = builder.locator("[data-custom-grid-count]").inner_text()
    warning_text = builder.locator("[data-custom-grid-warning]").inner_text()
    assert "2 cases" in count_text, warning_text
    assert builder.get_attribute("data-custom-grid-valid") == "true"
    config = builder.locator("[data-custom-grid-config]").input_value()
    assert '"anti_blaze_angle_deg":4.5' in config
    page.locator("select[name='workflow']").select_option("energy_sweep")
    scans = page.locator("[data-parameter-scans]")
    scans.locator("[data-custom-add-axis]").click()
    scan_axis = scans.locator("[data-custom-axis]").first
    scan_axis.locator(".custom-grid-parameter").select_option("blaze_angle_deg")
    scan_axis.locator("[data-axis-mode]").select_option("range")
    scan_axis.locator("[data-count]").fill("50")
    preview = scans.locator("[data-custom-grid-preview]")
    assert preview.evaluate("node => node.scrollHeight > node.clientHeight")
    summary_box = scans.locator("[data-custom-grid-summary]").bounding_box()
    axis_box = scans.locator(".custom-grid-section").first.bounding_box()
    assert summary_box and axis_box and abs(summary_box["height"] - axis_box["height"]) < 36


@pytest.mark.browser
def test_afm_fraction_spinner_uses_hundredths(browser_page: Page) -> None:
    """Fine tune AFM thresholds using the number field arrows."""
    page = browser_page
    page.goto(f"{page.url.rstrip('/')}/gratings/new")
    page.locator("select[name='grating_type']").select_option("afm")
    separation = page.locator("input[name='afm_min_separation']")
    separation.fill("0.4")
    separation.press("ArrowUp")
    expect(separation).to_have_value("0.41")
    prominence = page.locator("input[name='afm_min_prominence']")
    prominence.fill("0.1")
    prominence.press("ArrowUp")
    expect(prominence).to_have_value("0.11")


@pytest.mark.browser
def test_plot_loop_keeps_scan_checkbox_selection(website, browser_page: Page) -> None:
    """Changing loop parameters must not silently select a cleared scan."""
    base_url, data_dir = website
    _write_sweep(data_dir)
    page = browser_page
    page.goto(f"{base_url}/plots/new")
    page.locator("[data-run-picker]").select_option("sweep")
    run = page.locator("[data-plot-run-item]")
    first_scan = run.locator("[data-scan-grid] input[type=checkbox]").first
    first_scan.uncheck()
    run.locator("[data-add-loop]").click()
    run.locator("[data-loop-list] select").select_option("diffraction_order")
    expect(first_scan).not_to_be_checked()
    expect(page.locator("[data-plot-preview-loading]")).to_have_text("Ready")


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

    expect(page.locator("select[name='polarization']")).to_have_value("p")
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
