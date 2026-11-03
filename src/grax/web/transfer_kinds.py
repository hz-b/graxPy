"""What import and export need to know about each kind of saved object."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

from werkzeug.datastructures import MultiDict

from .common import _is_safe_id
from .files import write_json_atomically
from .grating_roughness import validate_roughness_geometry
from .jobs import LIVE_STATES
from .multilayer_design_studies import (
    STAGES,
    MultilayerDesignStudyStore,
    MultilayerGratingStore,
    attach_study_grating,
    build_design_config,
    grating_config_subset,
    pass_grating_config_to_studies,
    study_config_defaults,
)
from .multilayer_jobs import ORPHANED_STAGE_ERROR, _multilayer_design_job_key
from .persistence import (
    SCHEMA_VERSION,
    GratingStore,
    PGMSetupStore,
    PlaneMirrorStore,
    build_grating_from_spec,
)
from .pgm_jobs import grating_from_form, mirror_from_form, parse_pgm_form
from .plane_mirror_views import _plane_mirror_stack
from .run_state import _is_run_active
from .runs import RunStore

ORPHANED_RUN_ERROR = "The app stopped before this run finished."
_FINISHED_STATUSES = {"completed", "aborted", "failed"}
_LEGACY_STATUSES = {"ok": "completed", "paused": "aborted", "interrupted": "aborted", "error": "failed"}


class ObjectKind:
    """A saved object kind whose objects are JSON files and whose runs are run folders."""

    key = ""
    label = ""
    plural = ""
    manage_endpoint = ""
    object_folder = ""
    run_folder = "runs"
    link_key = ""
    link_name_key = ""
    name_key = "name"
    manifest_name = "manifest.json"
    store_class: type[GratingStore] = GratingStore

    def store(self, data_dir: Path) -> Any:
        return self.store_class(data_dir / self.object_folder)

    def list_objects(self, data_dir: Path) -> list[dict[str, Any]]:
        return self.store(data_dir).list()

    def load(self, data_dir: Path, object_id: str) -> dict[str, Any]:
        return self.store(data_dir).load(object_id)

    def object_name(self, spec: dict[str, Any]) -> str:
        return str(spec.get(self.name_key) or spec.get("id") or self.label)

    def set_object_name(self, spec: dict[str, Any], name: str) -> None:
        spec[self.name_key] = name

    def object_exists(self, data_dir: Path, object_id: str) -> bool:
        """Return whether the ID is taken, also by the files a deleted object left behind."""
        directory = data_dir / self.object_folder
        return (directory / f"{object_id}.json").exists() or (directory / object_id).exists()

    def sidecar_folder(self, data_dir: Path, object_id: str) -> Path:
        """Return the folder that holds the files saved beside the object."""
        return data_dir / self.object_folder / object_id

    def write_object(self, data_dir: Path, spec: dict[str, Any]) -> None:
        path = self.store(data_dir)._path_for_id(str(spec["id"]))
        write_json_atomically(path, spec, sort_keys=True)

    def remove_object(self, data_dir: Path, object_id: str) -> None:
        self.store(data_dir).delete(object_id)

    def stamp_saved_object(self, spec: dict[str, Any], now: str) -> None:
        """Record the current format and time, as the object's store does on save."""
        spec["schema_version"] = SCHEMA_VERSION
        spec["updated_at"] = now

    def schema_problem(self, spec: dict[str, Any]) -> str | None:
        version = spec.get("schema_version", 0)
        if isinstance(version, int) and version > SCHEMA_VERSION:
            return "This file was made by a newer version of GRAx. Update GRAx to import it."
        return None

    def validate_object(self, spec: dict[str, Any], data_dir: Path) -> None:
        raise NotImplementedError

    def run_store(self, data_dir: Path) -> Any:
        return RunStore(data_dir / self.run_folder)

    def run_dir(self, data_dir: Path, run_id: str) -> Path:
        return self.run_store(data_dir).run_dir(run_id)

    def run_exists(self, data_dir: Path, run_id: str) -> bool:
        return (data_dir / self.run_folder / run_id).exists()

    def linked_runs(self, data_dir: Path, object_id: str) -> list[dict[str, Any]]:
        """Return the manifests of the object's runs."""
        return [
            run for run in self.run_store(data_dir).list()
            if str(run.get(self.link_key) or "") == object_id
        ]

    def is_run_active(self, app: Any, manifest: dict[str, Any]) -> bool:
        return _is_run_active(app, str(manifest["id"]))

    def run_name(self, manifest: dict[str, Any]) -> str:
        return str(manifest.get("display_name") or manifest.get("id") or "")

    def run_workflow(self, manifest: dict[str, Any]) -> str:
        return str(manifest.get("workflow") or "")

    def run_status(self, manifest: dict[str, Any]) -> str:
        return str(manifest.get("status") or "")

    def finish_run_status(self, manifest: dict[str, Any]) -> None:
        """Record a run no live job owns any more as finished, in the exported copy."""
        status = str(manifest.get("status") or "")
        status = _LEGACY_STATUSES.get(status, status)
        if status in LIVE_STATES or status not in _FINISHED_STATUSES:
            status = "aborted"
            manifest["error_text"] = manifest.get("error_text") or ORPHANED_RUN_ERROR
        manifest["status"] = status

    def manifest_problem(self, manifest: dict[str, Any]) -> str | None:
        """Return why an imported manifest could not be listed, or None."""
        for key in ("status", "created_at", "workflow"):
            if not isinstance(manifest.get(key), str):
                return f"Its run record has no valid {key}."
        if not isinstance(manifest.get("artifacts", []), list):
            return "Its run record lists its result files incorrectly."
        return None

    def relink_run(self, manifest: dict[str, Any], *, object_id: str, object_name: str) -> None:
        manifest[self.link_key] = object_id
        manifest[self.link_name_key] = object_name

    def skips_run_file(self, relative: PurePosixPath) -> bool:
        """Return whether a run file is a cache that is not worth sending."""
        return False

    def replace_local_object(
        self, data_dir: Path, *, target: dict[str, Any], imported: dict[str, Any]
    ) -> dict[str, Any]:
        """Return the object to save when ``imported`` replaces ``target``."""
        return {**imported, "id": target["id"], "created_at": target.get("created_at")}

    def local_run_restorer(self, data_dir: Path, object_id: str) -> Callable[[], None]:
        """Return a function that undoes what replacing the object does to its local runs."""
        return lambda: None


class GratingKind(ObjectKind):
    key = "grating"
    label = "Grating"
    plural = "Gratings"
    manage_endpoint = "manage_gratings"
    object_folder = "saved_gratings"
    link_key = "grating_id"
    link_name_key = "grating_name"

    def validate_object(self, spec: dict[str, Any], data_dir: Path) -> None:
        validate_roughness_geometry(build_grating_from_spec(spec))

    def relink_run(self, manifest: dict[str, Any], *, object_id: str, object_name: str) -> None:
        super().relink_run(manifest, object_id=object_id, object_name=object_name)
        if isinstance(manifest.get("grating_spec"), dict):
            manifest["grating_spec"].update(id=object_id, name=object_name)


class PlaneMirrorKind(ObjectKind):
    key = "plane_mirror"
    label = "Plane mirror"
    plural = "Plane mirrors"
    manage_endpoint = "plane_mirror_index"
    object_folder = "saved_plane_mirrors"
    link_key = "mirror_id"
    link_name_key = "mirror_name"
    store_class = PlaneMirrorStore

    def validate_object(self, spec: dict[str, Any], data_dir: Path) -> None:
        if not isinstance(spec.get("stack_form"), dict):
            raise ValueError("The mirror has no layer stack.")
        _plane_mirror_stack(_form_values(spec["stack_form"]))
        scans = spec.get("scans", [])
        if not isinstance(scans, list):
            raise ValueError("The mirror's saved scans are invalid.")
        for scan in scans:
            if not isinstance(scan, dict) or not _is_safe_id(scan.get("id")):
                raise ValueError("The mirror's saved scans are invalid.")


class PGMKind(ObjectKind):
    key = "pgm"
    label = "PGM"
    plural = "PGMs"
    manage_endpoint = "pgm_setup_index"
    object_folder = "saved_pgms"
    run_folder = "pgm_runs"
    link_key = "pgm_id"
    link_name_key = "pgm_name"
    store_class = PGMSetupStore

    def validate_object(self, spec: dict[str, Any], data_dir: Path) -> None:
        for key in ("grating_form", "mirror_form", "parameters"):
            if not isinstance(spec.get(key), dict):
                raise ValueError(f"The PGM has no valid {key.replace('_', ' ')}.")
        grating_from_form(_form_values(spec["grating_form"]), data_dir)
        mirror_from_form(_form_values(spec["mirror_form"]))
        parse_pgm_form(_form_values(spec["parameters"]))

    def relink_run(self, manifest: dict[str, Any], *, object_id: str, object_name: str) -> None:
        super().relink_run(manifest, object_id=object_id, object_name=object_name)
        # These name the sender's saved optics; the run keeps its own snapshots of them.
        manifest["grating_id"] = None
        manifest["mirror_id"] = None

    def skips_run_file(self, relative: PurePosixPath) -> bool:
        return relative.parts[0] == "cache"


class MultilayerGratingKind(ObjectKind):
    key = "multilayer_grating"
    label = "Multilayer grating"
    plural = "Multilayer gratings"
    manage_endpoint = "multilayer_gratings_manage"
    object_folder = "multilayer_gratings"
    run_folder = "multilayer_designs"
    link_key = "grating_id"
    name_key = "display_name"
    manifest_name = "study.json"
    store_class = MultilayerGratingStore

    def object_exists(self, data_dir: Path, object_id: str) -> bool:
        return (data_dir / self.object_folder / object_id).exists()

    def write_object(self, data_dir: Path, spec: dict[str, Any]) -> None:
        self.store(data_dir).save(spec)

    def schema_problem(self, spec: dict[str, Any]) -> str | None:
        return None

    def stamp_saved_object(self, spec: dict[str, Any], now: str) -> None:
        """Multilayer gratings carry neither a format version nor a change time."""

    def validate_object(self, spec: dict[str, Any], data_dir: Path) -> None:
        if not isinstance(spec.get("config"), dict):
            raise ValueError("The multilayer grating has no configuration.")
        build_design_config(data_dir, {**study_config_defaults(), **spec["config"]})

    def run_store(self, data_dir: Path) -> MultilayerDesignStudyStore:
        return MultilayerDesignStudyStore(data_dir / self.run_folder)

    def run_dir(self, data_dir: Path, run_id: str) -> Path:
        return self.run_store(data_dir).study_dir(run_id)

    def linked_runs(self, data_dir: Path, object_id: str) -> list[dict[str, Any]]:
        studies = self.run_store(data_dir)
        gratings = self.store(data_dir)
        attached = [
            attach_study_grating(study, studies=studies, gratings=gratings)
            for study in studies.list()
        ]
        return [study for study in attached if str(study.get("grating_id")) == object_id]

    def is_run_active(self, app: Any, manifest: dict[str, Any]) -> bool:
        study_id = str(manifest["id"])
        return any(
            _is_run_active(app, _multilayer_design_job_key(study_id, stage)) for stage in STAGES
        )

    def run_workflow(self, manifest: dict[str, Any]) -> str:
        return "study"

    def run_status(self, manifest: dict[str, Any]) -> str:
        stages = manifest.get("stages") or {}
        return " · ".join(
            f"{stage.replace('_', ' ')}: {(stages.get(stage) or {}).get('status', 'not_run')}"
            for stage in STAGES
        )

    def finish_run_status(self, manifest: dict[str, Any]) -> None:
        for state in (manifest.get("stages") or {}).values():
            if state.get("status") in LIVE_STATES:
                state.update(status="aborted", error_text=ORPHANED_STAGE_ERROR)

    def manifest_problem(self, manifest: dict[str, Any]) -> str | None:
        stages = manifest.get("stages")
        if not isinstance(manifest.get("config"), dict) or not isinstance(stages, dict):
            return "Its study record is incomplete."
        for stage in STAGES:
            state = stages.get(stage)
            if not isinstance(state, dict) or not isinstance(state.get("status"), str):
                return "Its study record is incomplete."
        return None

    def relink_run(self, manifest: dict[str, Any], *, object_id: str, object_name: str) -> None:
        manifest["grating_id"] = object_id
        manifest.pop("family_id", None)

    def replace_local_object(
        self, data_dir: Path, *, target: dict[str, Any], imported: dict[str, Any]
    ) -> dict[str, Any]:
        """Pass the imported geometry to the local studies, as editing a grating does."""
        config = grating_config_subset(imported["config"])
        pass_grating_config_to_studies(
            self.linked_runs(data_dir, str(target["id"])), config, studies=self.run_store(data_dir)
        )
        return {**target, "display_name": imported["display_name"], "config": config}

    def local_run_restorer(self, data_dir: Path, object_id: str) -> Callable[[], None]:
        studies = self.run_store(data_dir)
        saved = self.linked_runs(data_dir, object_id)

        def restore() -> None:
            for study in saved:
                studies.save(study)

        return restore


def _form_values(fields: dict[str, Any]) -> MultiDict:
    """Return saved form fields as the submitted form they came from."""
    form = MultiDict()
    for key, value in fields.items():
        if isinstance(value, list):
            form.setlist(key, [str(item) for item in value])
        else:
            form[key] = "" if value is None else str(value)
    return form


KINDS: dict[str, ObjectKind] = {
    kind.key: kind
    for kind in (GratingKind(), PlaneMirrorKind(), PGMKind(), MultilayerGratingKind())
}
