"""Flask application for the local grax web MVP."""

from __future__ import annotations

import argparse
import json
import os
import shutil  # noqa: F401 - tests patch module functions through grax.web.app
import subprocess  # noqa: F401 - tests patch module functions through grax.web.app
import threading
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np  # noqa: F401 - kept importable from grax.web.app

# Re-exported so existing imports from grax.web.app keep working.
from .common import (  # noqa: F401, E402
    FRONTEND_SCRIPTS,
    PATH_ID_KEYS,
    _file_token,
    _grax_version,
    _is_cross_site_request,
    _is_loopback_address,
    _is_safe_id,
    _normalized_polarization,
    _normalized_solver,
    _site_metadata,
    _write_text_atomically,
)
from .context import WebContext
from .grating_forms import (  # noqa: F401
    _attach_roughness,
    _default_form_values,
    _grating_custom_layer_rows,
    _spec_from_form,
)
from .multilayer_jobs import (  # noqa: F401
    _energy_scan_checkpoint_progress,
    _energy_scan_checkpoint_series,
)
from .persistence import GratingStore
from .plot_builder import (  # noqa: F401
    _build_plotly_compare_payload,
    _compare_x_values,
    _load_selected_plot_series,
    _require_plotly,
    _save_compare_image,
    _series_controls_for_selected_series,
)
from .run_exports import _run_metadata_payload  # noqa: F401
from .run_files import (  # noqa: F401
    _available_orders,
    _load_checkpoint_results,
    _load_order_series,
    _load_run_manifest,
    _parameter_study_artifacts,
    _replace_plot_with_retry,
)
from .run_inputs import (  # noqa: F401
    _cases_for_workflow,
    _custom_grid_defaults_from_form,
    _custom_grid_from_form,
    _parameter_scan_point_count,
    _parameter_scans_from_form,
    _run_input_from_form,
    _snapshot_custom_grid_afm_profiles,
)
from .run_jobs import _queue_run, _start_run_worker  # noqa: F401
from .run_plots import _publish_live_progress_snapshot, _results_sorted_for_live_plot  # noqa: F401
from .run_state import ActiveRunState, psutil  # noqa: F401
from .run_status import _memory_status_payload  # noqa: F401
from .runs import RunStore
from .workspace import _pick_native_directory, platform  # noqa: F401


def create_app(*, data_dir: str | Path | None = None):
    """Create the local Flask app.

    Args:
        data_dir: Directory used for saved gratings, run artifacts, and previews.

    Returns:
        Configured Flask application.
    """
    try:
        from flask import Flask, abort, request
    except ModuleNotFoundError as error:  # pragma: no cover - exercised by user startup.
        raise RuntimeError(
            "The web app requires Flask. Install it with either "
            "python -m pip install \"graxpy[web]\" or, from a local checkout, "
            "python -m pip install -e \".[web]\""
        ) from error

    from .routes import core, gratings, multilayer, plane_mirrors, plots, runs

    app = Flask(__name__)
    app.config["GRAx_DATA_DIR"] = Path(data_dir or Path.cwd() / ".grax-web").resolve()

    @app.url_defaults
    def version_frontend_assets(endpoint: str, values: dict[str, Any]) -> None:
        """Load updated frontend assets after a page refresh."""
        filename = str(values.get("filename", ""))
        if endpoint == "static" and (filename.endswith((".js", ".css")) or filename in {
            "graxpy-logo.svg", "hzb-bessy2-logo.png", "graxpy-hzb-logo.png",
        }):
            values.setdefault("v", _file_token(Path(app.static_folder) / values["filename"]))

    @app.url_value_preprocessor
    def reject_unsafe_ids(endpoint: str | None, values: dict[str, Any] | None) -> None:
        """Answer 404 for URL IDs that could point outside the workspace folders."""
        for key in PATH_ID_KEYS:
            value = (values or {}).get(key)
            if value is not None and not _is_safe_id(value):
                abort(404)

    @app.before_request
    def reject_cross_site_writes() -> None:
        """Refuse state-changing requests that another website's page sent."""
        if request.method in {"GET", "HEAD", "OPTIONS"}:
            return
        if _is_cross_site_request(
            origin=request.headers.get("Origin"),
            fetch_site=request.headers.get("Sec-Fetch-Site"),
            host_url=request.host_url,
        ):
            abort(403, "Requests from other websites are not allowed.")

    def store() -> GratingStore:
        return GratingStore(app.config["GRAx_DATA_DIR"] / "saved_gratings")

    def run_store() -> RunStore:
        return RunStore(app.config["GRAx_DATA_DIR"] / "runs")

    def active_data_dir() -> Path:
        return Path(app.config["GRAx_DATA_DIR"]).resolve()

    def load_user_settings() -> dict[str, Any]:
        settings_path = active_data_dir() / "settings.json"
        try:
            with settings_path.open("r", encoding="utf-8") as handle:
                stored = json.load(handle)
        except (OSError, json.JSONDecodeError):
            stored = {}
        if not isinstance(stored, dict):
            stored = {}
        decimal_separator = stored.get("decimal_separator")
        standard_polarization = stored.get("standard_polarization")
        saved = decimal_separator in {"dot", "comma"} and standard_polarization in {"s", "p"}
        return {
            "decimal_separator": decimal_separator if decimal_separator in {"dot", "comma"} else "dot",
            "standard_polarization": standard_polarization if standard_polarization in {"s", "p"} else "p",
            "saved": saved,
        }

    @app.context_processor
    def inject_site_metadata() -> dict[str, Any]:
        """Inject shared site attribution and saved user settings into templates."""

        return {
            "site_metadata": _site_metadata(),
            "app_settings": load_user_settings(),
            "frontend_scripts": FRONTEND_SCRIPTS,
        }

    ctx = WebContext(
        active_data_dir=active_data_dir,
        load_user_settings=load_user_settings,
        run_store=run_store,
        store=store,
    )
    for group in (core, gratings, runs, plots, multilayer, plane_mirrors):
        group.register(app, ctx)

    return app


def main() -> None:
    """Run the local development server."""

    parser = argparse.ArgumentParser(prog="grax-web")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Database folder to serve (default: .grax-web in the current folder).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable the Flask debugger and auto-reloader (local hosts only; restarts stop running jobs).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the website in the default browser on start.",
    )
    args = parser.parse_args()
    if args.debug and not (args.host == "localhost" or _is_loopback_address(args.host)):
        parser.error("--debug exposes an interactive debugger and is only allowed with a local --host.")
    app = create_app() if args.data_dir is None else create_app(data_dir=args.data_dir)
    if not args.no_browser:
        _maybe_open_browser(args.host, args.port, reloader=args.debug)
    app.run(host=args.host, port=args.port, debug=args.debug)


def _maybe_open_browser(host: str, port: int, *, reloader: bool = False) -> None:
    """Open the local web UI in the default browser when appropriate.

    Args:
        host: Flask bind host.
        port: Flask bind port.
        reloader: Whether the server runs under the Werkzeug reloader, which
            starts the app twice; the browser then opens from the child only.
    """

    if reloader and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        return

    browser_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url = f"http://{browser_host}:{int(port)}"
    timer = threading.Timer(1.0, lambda: webbrowser.open(url))
    timer.daemon = True
    timer.start()
