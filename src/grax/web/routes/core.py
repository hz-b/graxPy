"""Home, guide, settings, workspace and raw data-file routes."""

from __future__ import annotations

import json
import subprocess
from typing import (
    Any,
)

from flask import (
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)

from ..catalog import (
    _list_plot_runs,
)
from ..common import (
    _is_loopback_address,
)
from ..context import WebContext
from ..run_jobs import (
    _memory_status_payload,
)
from ..run_state import (
    _has_active_runs,
)
from ..workspace import (
    _pick_native_directory,
    _validate_workspace_root,
)


def register(app: Any, ctx: WebContext) -> None:
    """Register the core routes on ``app``."""
    active_data_dir = ctx.active_data_dir
    store = ctx.store

    @app.post("/settings")
    def update_user_settings():
        """Persist display preferences beside the active database."""

        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            abort(400, "Settings must be sent as a JSON object.")
        decimal_separator = payload.get("decimal_separator")
        standard_polarization = payload.get("standard_polarization")
        if decimal_separator not in {"dot", "comma"}:
            abort(400, "Decimal separator must be dot or comma.")
        if standard_polarization not in {"s", "p"}:
            abort(400, "Standard polarization must be s or p.")
        settings_path = active_data_dir() / "settings.json"
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = settings_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps({
                "decimal_separator": decimal_separator,
                "standard_polarization": standard_polarization,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(settings_path)
        return jsonify({"saved": True})

    @app.get("/")
    def index():
        data_dir = active_data_dir()
        gratings = store().list()
        gratings.sort(
            key=lambda grating: (
                str(grating.get("updated_at", "")),
                str(grating.get("created_at", "")),
                str(grating.get("name", "")).lower(),
            ),
            reverse=True,
        )
        return render_template(
            "index.html",
            gratings=gratings,
            runs=_list_plot_runs(data_dir / "runs"),
            results_dir=data_dir / "runs",
            plots_dir=data_dir / "plots",
            active_workspace=data_dir,
            message=str(request.args.get("message", "")).strip(),
            message_kind=str(request.args.get("kind", "info")).strip() or "info",
        )

    @app.get("/docs")
    def web_docs():
        return render_template(
            "web_docs.html",
            workspace_root=active_data_dir(),
        )

    @app.get("/docs/<topic>")
    def web_docs_topic(topic: str):
        templates = {
            "gratings": "web_docs_gratings.html",
            "plane-mirrors": "web_docs_plane_mirrors.html",
            "multilayer-gratings": "web_docs_multilayer_gratings.html",
        }
        if topic not in templates:
            abort(404)
        return render_template(templates[topic])

    @app.post("/workspace/pick")
    def pick_workspace():
        if _has_active_runs(app):
            return redirect(
                url_for(
                    "index",
                    message="Cannot change the database folder while runs are active.",
                    kind="error",
                )
            )
        if not _is_loopback_address(request.remote_addr):
            return redirect(
                url_for(
                    "index",
                    message=(
                        "System folder selection is available only from the local machine."
                    ),
                    kind="error",
                )
            )
        try:
            selected_dir = _pick_native_directory()
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            return redirect(
                url_for(
                    "index",
                    message=(
                        f"The system folder picker is unavailable ({error}). "
                        "Use a local desktop session with a supported folder picker."
                    ),
                    kind="error",
                )
            )
        if selected_dir is None:
            return redirect(
                url_for(
                    "index",
                    message="Folder selection canceled. The active database folder was not changed.",
                )
            )
        try:
            target_dir = _validate_workspace_root(selected_dir)
        except (OSError, ValueError) as error:
            return redirect(url_for("index", message=str(error), kind="error"))
        app.config["GRAx_DATA_DIR"] = target_dir
        return redirect(
            url_for("index", message=f"Database folder set to {target_dir}", kind="success")
        )

    @app.get("/system/memory")
    def system_memory():
        return jsonify(_memory_status_payload())

    @app.get("/system/resource-status")
    def system_resource_status():
        from .resource_manager import resource_status
        return jsonify(resource_status())

    @app.get("/_data/<path:filename>")
    def data_file(filename: str):
        return send_from_directory(app.config["GRAx_DATA_DIR"], filename)
