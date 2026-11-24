"""Bring a saved database up to the current standard, with progress for the page.

Older databases lack the saved run summaries and preview images that make the
list pages fast. They are created on demand anyway, but this module does it all
in the background after opening, so the pages can show how many items are done
and how many are missing, and later remember that the database is up to date.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from .files import read_json, write_json_atomically
from .run_files import _cached_run_summary, _run_list_summary

logger = logging.getLogger(__name__)

# Raise this when a new saved item needs to be created for older databases.
DATABASE_STANDARD_VERSION = 1

_LOCK = threading.Lock()
_WORKERS: dict[str, "_Upgrade"] = {}


class _Upgrade:
    """Progress of one background upgrade of a database folder."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.processed = 0
        self.failed: list[str] = []
        self.finished = False


def _marker_path(data_dir: Path) -> Path:
    return data_dir / "cache" / "database_version.json"


def _read_marker(data_dir: Path) -> dict[str, Any]:
    try:
        marker = read_json(_marker_path(data_dir))
    except (OSError, ValueError):
        return {}
    return marker if isinstance(marker, dict) else {}


def _run_preview_path(data_dir: Path, run_id: str) -> Path:
    return data_dir / "previews" / "run-stacks" / f"{run_id}.png"


def _grating_preview_path(data_dir: Path, grating_id: str) -> Path:
    return data_dir / "previews" / f"{grating_id}.png"


def _items(data_dir: Path) -> list[tuple[str, str, dict[str, Any]]]:
    """List every saved run and grating as (kind, id, saved definition)."""
    from .persistence import GratingStore
    from .runs import RunStore

    items: list[tuple[str, str, dict[str, Any]]] = []
    for run in RunStore(data_dir / "runs").list():
        items.append(("run", str(run["id"]), run))
    for grating in GratingStore(data_dir / "saved_gratings").list():
        items.append(("grating", str(grating["id"]), grating))
    return items


def _is_current(data_dir: Path, kind: str, item_id: str, item: dict[str, Any]) -> bool:
    """Return whether one saved item already has everything the standard asks for."""
    if kind == "run":
        if _cached_run_summary(data_dir / "runs" / item_id) is None:
            return False
        has_spec = isinstance(item.get("grating_spec"), dict)
        return not has_spec or _run_preview_path(data_dir, item_id).is_file()
    return _grating_preview_path(data_dir, item_id).is_file()


def _upgrade_item(data_dir: Path, kind: str, item_id: str, item: dict[str, Any]) -> None:
    """Create what one saved item is missing."""
    from .catalog import _draw_run_stack_preview

    if kind == "run":
        _run_list_summary(data_dir / "runs" / item_id)
        spec = item.get("grating_spec")
        preview = _run_preview_path(data_dir, item_id)
        if isinstance(spec, dict) and not preview.is_file():
            _draw_run_stack_preview(spec, preview)
        return
    from .grating_forms import _render_grating_previews
    from .persistence import build_grating_from_spec
    from .run_plots import _LIVE_PLOT_RENDER_LOCK

    with _LIVE_PLOT_RENDER_LOCK:
        _render_grating_previews(build_grating_from_spec(item), _grating_preview_path(data_dir, item_id))


def _run_upgrade(data_dir: Path, state: _Upgrade) -> None:
    """Upgrade every item that is not current, one at a time."""
    try:
        failed_before = set(_read_marker(data_dir).get("failed", []))
        for kind, item_id, item in _items(data_dir):
            key = f"{kind}:{item_id}"
            if key in failed_before or _is_current(data_dir, kind, item_id, item):
                continue
            try:
                _upgrade_item(data_dir, kind, item_id, item)
            except Exception:
                logger.warning("Could not update %s %s to the current standard", kind, item_id, exc_info=True)
                state.failed.append(key)
            state.processed += 1
        marker = {
            "version": DATABASE_STANDARD_VERSION,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "failed": sorted(failed_before | set(state.failed)),
        }
        try:
            write_json_atomically(_marker_path(data_dir), marker)
        except OSError:
            logger.warning("Could not record the database version in %s", data_dir)
    finally:
        state.finished = True


def database_status(data_dir: Path, *, start: bool = True) -> dict[str, Any]:
    """Return how many saved items follow the current standard, starting the upgrade if needed.

    ``state`` is ``"updating"`` while the background upgrade runs and
    ``"current"`` once nothing is missing.
    """
    data_dir = Path(data_dir)
    marker = _read_marker(data_dir)
    known_failed = set(marker.get("failed", [])) if marker.get("version") == DATABASE_STANDARD_VERSION else set()
    items = _items(data_dir)
    key = str(data_dir.resolve())
    with _LOCK:
        worker = _WORKERS.get(key)
    skipped = known_failed | set(worker.failed if worker is not None else ())
    missing = [
        f"{kind}:{item_id}" for kind, item_id, item in items
        if f"{kind}:{item_id}" not in skipped and not _is_current(data_dir, kind, item_id, item)
    ]
    with _LOCK:
        worker = _WORKERS.get(key)
        running = worker is not None and not worker.finished
        if missing and not running and start:
            worker = _Upgrade(total=len(missing))
            _WORKERS[key] = worker
            threading.Thread(
                target=_run_upgrade, args=(data_dir, worker), daemon=True, name="database-upgrade"
            ).start()
            running = True
    total = worker.total if worker is not None and running else len(items)
    done = max(total - len(missing), 0)
    if not missing and not running:
        if marker.get("version") != DATABASE_STANDARD_VERSION:
            try:
                write_json_atomically(_marker_path(data_dir), {
                    "version": DATABASE_STANDARD_VERSION,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    "failed": sorted(known_failed),
                })
            except OSError:
                pass
        return {
            "state": "current", "total": len(items), "done": len(items), "missing": 0,
            "failed": len(known_failed), "version": DATABASE_STANDARD_VERSION,
        }
    return {
        "state": "updating", "total": total, "done": done, "missing": max(total - done, 0),
        "failed": len(worker.failed) if worker is not None else 0,
        "version": DATABASE_STANDARD_VERSION,
    }
