"""Carry a saved object's new name over to the runs made from it."""

from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from .runs import RunStore


def _rename_in_title(title: str, old_names: set[str], new_name: str) -> str:
    """Replace whole occurrences of the old names, so "Pt" leaves "Ptolemy" alone."""
    for old_name in sorted(filter(None, old_names), key=len, reverse=True):
        pattern = rf"(?<!\w){re.escape(old_name)}(?!\w)"
        title = re.sub(pattern, lambda _match: new_name, title)
    return title


def _rename_linked_runs(
    store: RunStore,
    *,
    link_key: str,
    name_key: str,
    snapshot_keys: tuple[str, ...],
    object_id: str,
    previous_name: str,
    new_name: str,
) -> None:
    """Update the stored object name and the title of every run linked to one object."""

    def change(run: dict[str, Any]) -> None:
        # A run's own copy may predate earlier renames, which this also corrects.
        old_names = {previous_name, str(run.get(name_key) or "")}
        run[name_key] = new_name
        for key in snapshot_keys:
            if isinstance(run.get(key), dict):
                run[key]["name"] = new_name
        run["display_name"] = _rename_in_title(
            str(run.get("display_name") or ""), old_names, new_name
        )

    for run in store.list():
        if str(run.get(link_key) or "") != object_id:
            continue
        renamed = deepcopy(run)
        change(renamed)
        if renamed != run:
            store.update(str(run["id"]), change)


def rename_grating_runs(data_dir: Path, grating_id: str, previous_name: str, new_name: str) -> None:
    """Rename a saved grating in its grating runs and the PGM runs that used it."""
    if new_name == previous_name:
        return
    for directory, snapshot_key in (("runs", "grating_spec"), ("pgm_runs", "grating_snapshot")):
        _rename_linked_runs(
            RunStore(Path(data_dir) / directory),
            link_key="grating_id",
            name_key="grating_name",
            snapshot_keys=(snapshot_key,),
            object_id=grating_id,
            previous_name=previous_name,
            new_name=new_name,
        )


def rename_mirror_runs(data_dir: Path, mirror_id: str, previous_name: str, new_name: str) -> None:
    """Rename a saved plane mirror in its runs and the PGM runs that used it."""
    if new_name == previous_name:
        return
    for directory in ("runs", "pgm_runs"):
        _rename_linked_runs(
            RunStore(Path(data_dir) / directory),
            link_key="mirror_id",
            name_key="mirror_name",
            snapshot_keys=("mirror_snapshot",),
            object_id=mirror_id,
            previous_name=previous_name,
            new_name=new_name,
        )


def rename_pgm_runs(data_dir: Path, pgm_id: str, previous_name: str, new_name: str) -> None:
    """Rename a saved PGM in the runs started from it."""
    if new_name == previous_name:
        return
    _rename_linked_runs(
        RunStore(Path(data_dir) / "pgm_runs"),
        link_key="pgm_id",
        name_key="pgm_name",
        snapshot_keys=(),
        object_id=pgm_id,
        previous_name=previous_name,
        new_name=new_name,
    )


def rename_multilayer_studies(
    store: Any, grating_id: str, previous_name: str, new_name: str
) -> None:
    """Rename a multilayer grating in the titles of its studies.

    Pages look up the grating's own name when they render, so only the titles hold a copy.
    """
    if new_name == previous_name:
        return

    def change(study: dict[str, Any]) -> None:
        study["display_name"] = _rename_in_title(
            str(study.get("display_name") or ""), {previous_name}, new_name
        )

    for study in store.list():
        if str(study.get("grating_id") or "") != grating_id:
            continue
        title = str(study.get("display_name") or "")
        if _rename_in_title(title, {previous_name}, new_name) != title:
            store.update(str(study["id"]), change)
