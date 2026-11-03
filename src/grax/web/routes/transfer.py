"""Export and import of saved objects with their runs.

These routes answer the page's scripts in JSON, so a failure appears on the page
the user is on. Nothing that goes wrong during an import may become a server error.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from functools import wraps
from typing import Any

from flask import (
    abort,
    jsonify,
    render_template,
    request,
    send_file,
    url_for,
)
from werkzeug.exceptions import HTTPException

from ..context import WebContext
from ..transfer import (
    TransferError,
    cancel_import,
    export_runs,
    import_preview,
    install_package,
    kind_or_none,
    package_filename,
    stage_upload,
    upload_too_large,
    write_package,
)
from ..transfer_kinds import KINDS, ObjectKind


def register(app: Any, ctx: WebContext) -> None:
    """Register the export and import routes on ``app``."""

    active_data_dir = ctx.active_data_dir

    def _answers_errors_in_json(view: Callable[..., Any]) -> Callable[..., Any]:
        """Turn every failure of a view into a message for the page that called it."""

        @wraps(view)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return view(*args, **kwargs)
            except TransferError as error:
                return jsonify({"error": str(error), "can_retry": error.can_retry}), 422
            except HTTPException as error:
                return jsonify({"error": error.description or "This request failed."}), error.code
            except Exception as error:
                app.logger.exception("An export or import request failed.")
                return jsonify({"error": f"This could not be completed: {error}"}), 500

        return wrapper

    def _kind_or_404(kind: str) -> ObjectKind:
        found = kind_or_none(kind)
        if found is None:
            abort(404, "This kind of object does not exist.")
        return found

    def _object_or_404(kind: ObjectKind, object_id: str) -> dict[str, Any]:
        try:
            return kind.load(active_data_dir(), object_id)
        except (ValueError, FileNotFoundError, OSError):
            abort(404, f"This {kind.label.lower()} no longer exists.")

    @app.get("/transfer/objects")
    @_answers_errors_in_json
    def transfer_objects():
        """List every object that can be exported, for the home page's dialog."""

        data_dir = active_data_dir()
        return jsonify([
            {
                "kind": kind.key,
                "label": kind.plural,
                "objects": [
                    {
                        "id": str(spec["id"]),
                        "name": kind.object_name(spec),
                        "runs_url": url_for("transfer_object_runs", kind=kind.key, object_id=spec["id"]),
                        "export_url": url_for("transfer_export", kind=kind.key, object_id=spec["id"]),
                    }
                    for spec in kind.list_objects(data_dir)
                ],
            }
            for kind in KINDS.values()
        ])

    @app.get("/transfer/<kind>/<object_id>/runs")
    @_answers_errors_in_json
    def transfer_object_runs(kind: str, object_id: str):
        object_kind = _kind_or_404(kind)
        spec = _object_or_404(object_kind, object_id)
        return jsonify({
            "name": object_kind.object_name(spec),
            "filename": package_filename(object_kind, spec),
            "runs": export_runs(app, object_kind, active_data_dir(), object_id),
        })

    @app.post("/transfer/<kind>/<object_id>/export")
    @_answers_errors_in_json
    def transfer_export(kind: str, object_id: str):
        object_kind = _kind_or_404(kind)
        _object_or_404(object_kind, object_id)
        package = tempfile.TemporaryFile()
        try:
            spec = write_package(
                app, object_kind, active_data_dir(), object_id, request.form.getlist("run_id"), package
            )
            package.seek(0)
        except BaseException:
            package.close()
            raise
        return send_file(
            package,
            mimetype="application/zip",
            as_attachment=True,
            download_name=package_filename(object_kind, spec),
        )

    @app.post("/transfer/import")
    @_answers_errors_in_json
    def transfer_import_upload():
        if upload_too_large(request.content_length):
            raise TransferError("This file is too large to import.")
        upload = request.files.get("package")
        if upload is None or not upload.filename:
            raise TransferError("Choose a file to import.")
        token = stage_upload(active_data_dir(), upload.stream, request.content_length)
        return jsonify({"url": url_for("transfer_import_preview", token=token)})

    @app.get("/transfer/import/<token>")
    def transfer_import_preview(token: str):
        """Show what a file holds; a bad or expired token explains itself on this page."""

        try:
            preview = import_preview(active_data_dir(), token)
        except Exception as error:
            if not isinstance(error, TransferError):
                app.logger.exception("An import preview failed.")
            return render_template("transfer_import.html", preview=None, message=str(error)), 410
        return render_template("transfer_import.html", preview=preview, message="")

    @app.post("/transfer/import/<token>")
    @_answers_errors_in_json
    def transfer_import_commit(token: str):
        result = install_package(
            app,
            active_data_dir(),
            token,
            run_ids=request.form.getlist("run_id"),
            mode=str(request.form.get("mode", "copy")),
            target_id=str(request.form.get("target_id", "")),
        )
        runs = f"{result['run_count']} run{'s' if result['run_count'] != 1 else ''}"
        verb = "Replaced" if result["replaced"] else "Imported"
        return jsonify({
            "url": url_for(
                result["manage_endpoint"],
                message=f"{verb} {result['kind_label'].lower()} {result['name']} with {runs}.",
                kind="success",
            ),
        })

    @app.post("/transfer/import/<token>/cancel")
    @_answers_errors_in_json
    def transfer_import_cancel(token: str):
        cancel_import(active_data_dir(), token)
        return jsonify({"url": url_for("index")})
