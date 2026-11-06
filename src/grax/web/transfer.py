"""Exchange saved objects and their runs between GRAx installations as zip packages."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import time
import zipfile
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime
from functools import lru_cache
from hashlib import sha1
from pathlib import Path, PurePosixPath
from typing import IO, Any
from uuid import uuid4

from .common import _grax_version, _is_safe_id
from .files import (
    locked,
    move_tree,
    read_json,
    remove_tree,
    replace_file,
    write_json_atomically,
)
from .persistence import _slugify
from .transfer_kinds import KINDS, ObjectKind

FORMAT_NAME = "grax-package"
FORMAT_VERSION = 1
PACKAGE_FILE = "grax_package.json"
OBJECT_FILE = "object.json"
PACKAGE_PREFIX = "grax-package://"
_PATH_KEYS = {"profile_path", "afm_profile_path"}
_WORKSPACE_PLACEHOLDER = "<workspace>"

MAX_UPLOAD_BYTES = 4 * 1024**3
MAX_UNPACKED_BYTES = 16 * 1024**3
MAX_MEMBERS = 200_000
MAX_COMPRESSION_RATIO = 500
_RATIO_ALLOWANCE_BYTES = 64 * 1024**2
STAGE_LIFETIME_SECONDS = 24 * 3600
_FORBIDDEN_NAME_CHARACTERS = set('\\:<>"|?*')
_MISSING_PROFILE_PROBLEM = "Its AFM profile file is missing."
_RUNNING_PROBLEM = "It is still running."
# Every file in the database folder is served at /_data/, so a package must not bring
# in anything a browser would run. These are the only types runs are known to write.
_RUN_FILE_SUFFIXES = {".png", ".csv", ".dat", ".json", ".jsonl", ".npz", ".txt"}
_OBJECT_FILES = {OBJECT_FILE, "profile.csv", "source_upload"}
_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


class TransferError(ValueError):
    """An export or import that cannot go ahead, with a message for the user."""

    def __init__(self, message: str, *, can_retry: bool = True) -> None:
        """Say whether the same upload can still be imported after this failure."""
        super().__init__(message)
        self.can_retry = can_retry


def upload_too_large(content_length: int | None) -> bool:
    """Return whether a request is declared larger than an import may be."""
    return (content_length or 0) > MAX_UPLOAD_BYTES


def size_label(size_bytes: int) -> str:
    """Return a size for people to read."""
    size = float(size_bytes)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def kind_or_none(key: str) -> ObjectKind | None:
    """Return the object kind for a URL segment."""
    return KINDS.get(key)


def _rewrite(value: Any, on_path: Callable[[str], str], on_text: Callable[[str], str]) -> Any:
    """Return ``value`` with profile paths and other strings passed through the callbacks."""
    if isinstance(value, dict):
        return {
            key: (
                on_path(item) if key in _PATH_KEYS and isinstance(item, str)
                else _rewrite(item, on_path, on_text)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rewrite(item, on_path, on_text) for item in value]
    if isinstance(value, str):
        return on_text(value)
    return value


def _member_problem(name: str) -> str | None:
    """Return why an archive member's path cannot be unpacked safely, or None."""
    if not name or name.startswith("/") or "\x00" in name:
        return "an absolute path"
    if _FORBIDDEN_NAME_CHARACTERS & set(name):
        return "a path with reserved characters"
    if any(part in {"", ".", ".."} or part != part.rstrip(" .") for part in name.split("/")):
        return "a path that leaves its folder"
    if any(part.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES for part in name.split("/")):
        return "a file name Windows reserves"
    return None


def _skips_file(kind: ObjectKind, relative: PurePosixPath) -> bool:
    """Return whether a run folder's file stays out of the package."""
    if relative.name.endswith((".tmp", ".tmp.png")) or relative.suffix.lower() not in _RUN_FILE_SUFFIXES:
        return True
    if any(part.startswith(".deleting-") for part in relative.parts):
        return True
    return kind.skips_run_file(relative) or _member_problem(relative.as_posix()) is not None


def _run_files(kind: ObjectKind, run_dir: Path) -> list[tuple[Path, PurePosixPath]]:
    """Return the regular files of a run that belong in a package."""
    files = []
    for folder, directories, names in os.walk(run_dir, followlinks=False):
        directories.sort()
        for name in sorted(names):
            path = Path(folder) / name
            if path.is_symlink() or not path.is_file():
                continue
            relative = PurePosixPath(path.relative_to(run_dir).as_posix())
            if not _skips_file(kind, relative):
                files.append((path, relative))
    return files


def _missing_profile(manifest: dict[str, Any]) -> bool:
    """Return whether a manifest needs an AFM profile that is no longer on disk."""
    missing = False

    def check(value: str) -> str:
        nonlocal missing
        if value and not Path(value).is_file():
            missing = True
        return value

    _rewrite(manifest, check, lambda text: text)
    return missing


def _run_row(app: Any, kind: ObjectKind, data_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Return one line of the export dialog's run list."""
    run_id = str(manifest["id"])
    run_dir = kind.run_dir(data_dir, run_id)
    problem = None
    if kind.is_run_active(app, manifest):
        problem = _RUNNING_PROBLEM
    elif _missing_profile(manifest):
        problem = _MISSING_PROFILE_PROBLEM
    copy = deepcopy(manifest)
    if problem != _RUNNING_PROBLEM:
        kind.finish_run_status(copy)
    size_bytes = sum(path.stat().st_size for path, _ in _run_files(kind, run_dir))
    return {
        "id": run_id,
        "name": kind.run_name(manifest),
        "workflow": kind.run_workflow(manifest),
        "status": kind.run_status(copy),
        "created_at": str(manifest.get("created_at") or ""),
        "size_bytes": size_bytes,
        "size_label": size_label(size_bytes),
        "problem": problem,
    }


def export_runs(app: Any, kind: ObjectKind, data_dir: Path, object_id: str) -> list[dict[str, Any]]:
    """Return the runs that can be offered for one object's export."""
    return [
        _run_row(app, kind, data_dir, manifest)
        for manifest in kind.linked_runs(data_dir, object_id)
    ]


def package_filename(kind: ObjectKind, spec: dict[str, Any]) -> str:
    """Return the suggested file name for an object's package."""
    return f"{_slugify(kind.object_name(spec)) or kind.key.replace('_', '-')}.grax.zip"


class _ProfilePacker:
    """Turn absolute AFM profile paths into package paths while collecting the files."""

    def __init__(self, owner: Path | None, folder_name: str) -> None:
        self.owner = owner.resolve() if owner else None
        self.folder_name = folder_name
        self.files: dict[str, Path] = {}

    def add(self, value: str, *, required: bool) -> str:
        if not value:
            return value
        path = Path(value)
        if not path.is_file():
            if required:
                raise TransferError(_MISSING_PROFILE_PROBLEM)
            return ""
        resolved = path.resolve()
        if self.owner and resolved.is_relative_to(self.owner):
            return f"{PACKAGE_PREFIX}{resolved.relative_to(self.owner).as_posix()}"
        digest = sha1(str(resolved).encode("utf-8")).hexdigest()[:16]
        relative = f"{self.folder_name}/{digest}.csv"
        self.files[relative] = resolved
        return f"{PACKAGE_PREFIX}{relative}"


def _scrubber(data_dir: Path) -> Callable[[str], str]:
    """Return a function that hides the local database folder in saved text."""
    root = str(data_dir.resolve())
    return lambda text: text.replace(root, _WORKSPACE_PLACEHOLDER)


def _write_run_json(
    archive: zipfile.ZipFile,
    path: Path,
    member: str,
    packer: _ProfilePacker,
    scrub: Callable[[str], str],
    markers: tuple[bytes, ...],
) -> None:
    """Copy a run's JSON or JSONL file, without its local paths."""
    raw = path.read_bytes()
    if not any(marker in raw for marker in markers):
        archive.writestr(member, raw)
        return
    on_path = lambda value: packer.add(value, required=False)  # noqa: E731
    if path.suffix == ".jsonl":
        lines = []
        for line in raw.decode("utf-8", "replace").splitlines():
            try:
                lines.append(json.dumps(_rewrite(json.loads(line), on_path, scrub)))
            except json.JSONDecodeError:
                if not any(marker.decode() in line for marker in markers):
                    lines.append(line)
        archive.writestr(member, "\n".join(lines) + "\n")
        return
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return
    archive.writestr(member, json.dumps(_rewrite(payload, on_path, scrub), indent=2))


def _write_run(
    archive: zipfile.ZipFile,
    kind: ObjectKind,
    data_dir: Path,
    manifest: dict[str, Any],
) -> None:
    """Add one run folder to a package."""
    run_id = str(manifest["id"])
    run_dir = kind.run_dir(data_dir, run_id)
    prefix = f"runs/{run_id}"
    packer = _ProfilePacker(run_dir, "afm_profiles")
    scrub = _scrubber(data_dir)
    copy = deepcopy(manifest)
    kind.finish_run_status(copy)
    clean_manifest = _rewrite(copy, lambda value: packer.add(value, required=True), scrub)
    archive.writestr(f"{prefix}/{kind.manifest_name}", json.dumps(clean_manifest, indent=2))
    root = str(data_dir.resolve())
    markers = (b"profile_path", root.encode("utf-8"), json.dumps(root)[1:-1].encode("utf-8"))
    for path, relative in _run_files(kind, run_dir):
        member = f"{prefix}/{relative.as_posix()}"
        if relative.as_posix() == kind.manifest_name:
            continue
        if path.suffix in {".json", ".jsonl"}:
            _write_run_json(archive, path, member, packer, scrub, markers)
        elif path.suffix == ".txt":
            archive.writestr(member, scrub(path.read_text(encoding="utf-8", errors="replace")))
        else:
            archive.write(path, member)
    for relative, source in packer.files.items():
        archive.write(source, f"{prefix}/{relative}")


def _write_object(
    archive: zipfile.ZipFile, kind: ObjectKind, data_dir: Path, spec: dict[str, Any]
) -> None:
    """Add the object and the files saved beside it to a package."""
    sidecars: dict[str, Path] = {}

    def add_profile(value: str) -> str:
        if not value:
            return value
        path = Path(value)
        if not path.is_file():
            raise TransferError(_MISSING_PROFILE_PROBLEM)
        sidecars["profile.csv"] = path
        source_upload = path.parent / "source_upload"
        if source_upload.is_file():
            sidecars["source_upload"] = source_upload
        return f"{PACKAGE_PREFIX}profile.csv"

    clean = _rewrite(spec, add_profile, _scrubber(data_dir))
    archive.writestr(f"object/{OBJECT_FILE}", json.dumps(clean, indent=2))
    for name, source in sidecars.items():
        archive.write(source, f"object/{name}")


def write_package(
    app: Any,
    kind: ObjectKind,
    data_dir: Path,
    object_id: str,
    run_ids: list[str],
    handle: IO[bytes],
) -> dict[str, Any]:
    """Write one object and the selected runs into ``handle`` and return the object."""
    spec = kind.load(data_dir, object_id)
    manifests = {str(run["id"]): run for run in kind.linked_runs(data_dir, object_id)}
    rows = []
    for run_id in dict.fromkeys(run_ids):
        manifest = manifests.get(run_id)
        if manifest is None:
            raise TransferError(f"Run {run_id} does not belong to this object.")
        row = _run_row(app, kind, data_dir, manifest)
        if row["problem"]:
            raise TransferError(f"{row['name']} cannot be exported. {row['problem']}")
        rows.append(row)
    with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        archive.writestr(PACKAGE_FILE, json.dumps(
            {
                "format": FORMAT_NAME,
                "format_version": FORMAT_VERSION,
                "kind": kind.key,
                "grax_version": _grax_version(),
                "exported_at": datetime.now().isoformat(timespec="seconds"),
                "object": {"id": object_id, "name": kind.object_name(spec)},
                "runs": [
                    {key: row[key] for key in row if key not in {"problem", "size_label"}}
                    for row in rows
                ],
            },
            indent=2,
        ))
        _write_object(archive, kind, data_dir, spec)
        for run_id in dict.fromkeys(run_ids):
            _write_run(archive, kind, data_dir, manifests[run_id])
    return spec


def _imports_dir(data_dir: Path) -> Path:
    return data_dir / ".imports"


def _stage_dir(data_dir: Path, token: str) -> Path:
    if not _is_safe_id(token):
        raise TransferError("This import is no longer available. Choose the file again.", can_retry=False)
    return _imports_dir(data_dir) / token


def _purge_stale_stages(data_dir: Path) -> None:
    """Remove uploads that were never imported or cancelled."""
    folder = _imports_dir(data_dir)
    if not folder.is_dir():
        return
    deadline = time.time() - STAGE_LIFETIME_SECONDS
    for stage in folder.iterdir():
        try:
            if stage.is_dir() and stage.stat().st_mtime < deadline:
                remove_tree(stage)
        except OSError:
            continue


def _save_upload(upload: IO[bytes], target: Path) -> None:
    """Save an uploaded stream to disk, refusing one larger than the limit."""
    written = 0
    with target.open("wb") as handle:
        while chunk := upload.read(1024 * 1024):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                raise TransferError("This file is too large to import.")
            handle.write(chunk)


def _check_members(archive: zipfile.ZipFile, package_size: int, data_dir: Path) -> list[zipfile.ZipInfo]:
    """Refuse an archive that could escape its folder or fill the disk; return its files."""
    infos = archive.infolist()
    if len(infos) > MAX_MEMBERS:
        raise TransferError("This file holds too many files to import.")
    seen: set[str] = set()
    members = []
    for info in infos:
        name = info.filename
        if (problem := _member_problem(name.rstrip("/"))) is not None:
            raise TransferError(f"This file is not a GRAx package: it contains {problem}.")
        mode = info.external_attr >> 16
        if stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR}:
            raise TransferError("This file is not a GRAx package: it contains a link or special file.")
        if info.flag_bits & 0x1:
            raise TransferError("This file is encrypted and cannot be imported.")
        if info.is_dir():
            continue
        parts = name.split("/")
        allowed = (
            name == PACKAGE_FILE
            or (len(parts) == 2 and parts[0] == "object" and parts[1] in _OBJECT_FILES)
            or (len(parts) > 2 and parts[0] == "runs" and PurePosixPath(name).suffix.lower() in _RUN_FILE_SUFFIXES)
        )
        if not allowed:
            raise TransferError(f"This file is not a GRAx package: unexpected entry {name}.")
        folded = name.casefold()
        if folded in seen:
            raise TransferError(f"This file is not a GRAx package: duplicate entry {name}.")
        seen.add(folded)
        members.append(info)
    total = sum(info.file_size for info in members)
    if total > MAX_UNPACKED_BYTES:
        raise TransferError("This file is too large to unpack.")
    if total > package_size * MAX_COMPRESSION_RATIO + _RATIO_ALLOWANCE_BYTES:
        raise TransferError("This file unpacks to an implausible size and was refused.")
    if total > shutil.disk_usage(data_dir).free:
        raise TransferError("There is not enough free disk space to import this file.")
    return members


def _extract(archive: zipfile.ZipFile, members: list[zipfile.ZipInfo], target: Path) -> None:
    """Unpack checked members, counting the bytes since declared sizes can lie."""
    remaining = sum(info.file_size for info in members)
    for info in members:
        destination = target.joinpath(*info.filename.split("/"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source, destination.open("wb") as output:
            while chunk := source.read(1024 * 1024):
                remaining -= len(chunk)
                if remaining < 0:
                    raise TransferError("This file unpacks to more data than it declares.")
                output.write(chunk)


def _read_header(folder: Path) -> tuple[dict[str, Any], ObjectKind]:
    """Read and check the package description."""
    try:
        header = read_json(folder / PACKAGE_FILE)
    except (OSError, ValueError) as error:
        raise TransferError("This file is not a GRAx package: its description is missing.") from error
    if not isinstance(header, dict) or header.get("format") != FORMAT_NAME:
        raise TransferError("This file is not a GRAx package.")
    version = header.get("format_version")
    if not isinstance(version, int) or version < 1:
        raise TransferError("This file is not a GRAx package: its version is invalid.")
    if version > FORMAT_VERSION:
        raise TransferError("This file was made by a newer version of GRAx. Update GRAx to import it.")
    kind = KINDS.get(str(header.get("kind")))
    if kind is None:
        raise TransferError("This file holds a kind of object this version of GRAx does not know.")
    described = header.get("object")
    if not isinstance(described, dict) or not _is_safe_id(described.get("id")):
        raise TransferError("This file is not a GRAx package: its object is invalid.")
    runs = header.get("runs")
    if not isinstance(runs, list) or not all(
        isinstance(run, dict) and _is_safe_id(run.get("id")) for run in runs
    ):
        raise TransferError("This file is not a GRAx package: its run list is invalid.")
    return header, kind


def _restore_path(value: str, final_dir: Path, source_dir: Path) -> str:
    """Return a package path's location on this computer; any other path is refused."""
    if not value:
        return value
    if not value.startswith(PACKAGE_PREFIX):
        raise TransferError("It refers to a file outside the package.")
    relative = value.removeprefix(PACKAGE_PREFIX)
    if _member_problem(relative) is not None or not source_dir.joinpath(*relative.split("/")).is_file():
        raise TransferError("It refers to a file the package does not contain.")
    return str(final_dir.joinpath(*relative.split("/")))


def _restored_object(folder: Path, final_dir: Path) -> dict[str, Any]:
    """Load the package's object with its profile path pointing at ``final_dir``."""
    spec = read_json(folder / "object" / OBJECT_FILE)
    if not isinstance(spec, dict):
        raise ValueError("Its record is invalid.")

    def restore(value: str) -> str:
        if value and value != f"{PACKAGE_PREFIX}profile.csv":
            raise TransferError("It refers to a file outside the package.")
        return _restore_path(value, final_dir, folder / "object")

    return _rewrite(spec, restore, lambda text: text)


def _check_object(kind: ObjectKind, folder: Path, data_dir: Path) -> dict[str, Any]:
    """Validate the package's object, refusing the whole file if it is unusable."""
    try:
        spec = _restored_object(folder, folder / "object")
        if (problem := kind.schema_problem(spec)) is not None:
            raise TransferError(problem)
        kind.validate_object(spec, data_dir)
    except TransferError:
        raise
    except Exception as error:
        raise TransferError(f"The {kind.label.lower()} in this file is invalid: {error}") from error
    return spec


def _unsafe_artifact(value: Any) -> bool:
    """Return whether a run's recorded results name a location outside its folder."""
    if isinstance(value, dict):
        return any(_unsafe_artifact(item) for item in value.values())
    if isinstance(value, list):
        return any(_unsafe_artifact(item) for item in value)
    if isinstance(value, str):
        path = value.replace("\\", "/")
        return path.startswith("/") or bool(re.match(r"[A-Za-z]:", path)) or ".." in path.split("/")
    return False


def _rewrite_run_files(run_dir: Path, final_dir: Path) -> None:
    """Point a staged run's checkpoint files at the run's final folder."""
    on_path = lambda value: _restore_path(value, final_dir, run_dir)  # noqa: E731
    keep_text = lambda text: text  # noqa: E731
    for path in run_dir.rglob("*"):
        if path.suffix not in {".json", ".jsonl"} or not path.is_file():
            continue
        raw = path.read_text(encoding="utf-8", errors="replace")
        if "profile_path" not in raw:
            continue
        if path.suffix == ".jsonl":
            text = "\n".join(
                json.dumps(_rewrite(json.loads(line), on_path, keep_text))
                for line in raw.splitlines() if line.strip()
            ) + "\n"
        else:
            text = json.dumps(_rewrite(json.loads(raw), on_path, keep_text), indent=2)
        path.write_text(text, encoding="utf-8")


def _run_problem(kind: ObjectKind, folder: Path, run_id: str) -> str | None:
    """Return why a package's run cannot be imported, or None."""
    run_dir = folder / "runs" / run_id
    try:
        manifest = read_json(run_dir / kind.manifest_name)
        if not isinstance(manifest, dict) or manifest.get("id") != run_id:
            return "Its run record is invalid."
        if (problem := kind.manifest_problem(manifest)) is not None:
            return problem
        artifacts = [manifest.get("artifacts")]
        artifacts += [
            [state.get("artifacts"), state.get("designs")]
            for state in (manifest.get("stages") or {}).values() if isinstance(state, dict)
        ]
        if _unsafe_artifact(artifacts):
            return "It names result files outside its own folder."
        keep_text = lambda text: text  # noqa: E731
        on_path = lambda value: _restore_path(value, run_dir, run_dir)  # noqa: E731
        _rewrite(manifest, on_path, keep_text)
        for path in run_dir.rglob("*"):
            if path.name == kind.manifest_name or path.suffix not in {".json", ".jsonl"} or not path.is_file():
                continue
            raw = path.read_text(encoding="utf-8", errors="replace")
            if "profile_path" not in raw:
                continue
            for line in raw.splitlines() if path.suffix == ".jsonl" else [raw]:
                if line.strip():
                    _rewrite(json.loads(line), on_path, keep_text)
    except TransferError as error:
        return str(error)
    except Exception:
        return "Its run record is invalid."
    return None


def _resume_depends_on_paths(folder: Path, run_id: str) -> bool:
    """Return whether a run's checkpoints name AFM files, whose new location blocks resuming."""
    try:
        metadata = folder / "runs" / run_id / "checkpoints" / "metadata.json"
        return PACKAGE_PREFIX in metadata.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def _matching_objects(kind: ObjectKind, data_dir: Path, header: dict[str, Any]) -> list[dict[str, str]]:
    """Return local objects that the package's object may be an update of."""
    wanted_id = str(header["object"]["id"])
    wanted_name = str(header["object"].get("name") or "").strip().casefold()
    return [
        {"id": str(spec["id"]), "name": kind.object_name(spec)}
        for spec in kind.list_objects(data_dir)
        if str(spec.get("id")) == wanted_id
        or kind.object_name(spec).strip().casefold() == wanted_name
    ]


def _discard_stage(stage: Path) -> None:
    """Remove an upload; one that cannot be removed yet is purged by a later upload."""
    try:
        remove_tree(stage)
    except OSError:
        pass


def _unpack_package(stage: Path, data_dir: Path) -> Path:
    """Unpack a staged package into a fresh folder and return the folder."""
    package = stage / "package.zip"
    folder = stage / "extract"
    try:
        with zipfile.ZipFile(package) as archive:
            members = _check_members(archive, package.stat().st_size, data_dir)
            _extract(archive, members, folder)
    except TransferError:
        raise
    except OSError as error:
        raise TransferError(f"This file could not be unpacked here: {error}") from error
    except Exception as error:
        raise TransferError("This file is damaged and cannot be imported.") from error
    return folder


def stage_upload(data_dir: Path, upload: IO[bytes], content_length: int | None) -> str:
    """Check an uploaded package, unpack it aside and return the token of its preview."""
    if upload_too_large(content_length):
        raise TransferError("This file is too large to import.")
    _purge_stale_stages(data_dir)
    token = uuid4().hex
    stage = _stage_dir(data_dir, token)
    try:
        stage.mkdir(parents=True)
        package = stage / "package.zip"
        _save_upload(upload, package)
        if not zipfile.is_zipfile(package):
            raise TransferError("This file is not a GRAx package.")
        folder = _unpack_package(stage, data_dir)
        header, kind = _read_header(folder)
        _check_object(kind, folder, data_dir)
        listed = {str(run["id"]) for run in header["runs"]}
        runs_folder = folder / "runs"
        found = {path.name for path in runs_folder.iterdir()} if runs_folder.is_dir() else set()
        if not found <= listed:
            raise TransferError("This file is not a GRAx package: it holds unlisted runs.")
        write_json_atomically(stage / "preview.json", {
            "header": header,
            "problems": {
                run_id: (
                    _run_problem(kind, folder, run_id) if run_id in found
                    else "The package does not contain this run."
                )
                for run_id in listed
            },
            "cannot_resume": [
                run_id for run_id in sorted(found) if _resume_depends_on_paths(folder, run_id)
            ],
        })
    except TransferError:
        _discard_stage(stage)
        raise
    except Exception as error:
        _discard_stage(stage)
        raise TransferError(f"This file could not be read: {error}") from error
    return token


def _load_stage(data_dir: Path, token: str) -> tuple[dict[str, Any], Path]:
    stage = _stage_dir(data_dir, token)
    try:
        return read_json(stage / "preview.json"), stage
    except (OSError, ValueError) as error:
        raise TransferError(
            "This import is no longer available. Choose the file again.", can_retry=False
        ) from error


def import_preview(data_dir: Path, token: str) -> dict[str, Any]:
    """Return what the preview page shows for a staged package."""
    preview, _stage = _load_stage(data_dir, token)
    header = preview["header"]
    kind = KINDS[header["kind"]]
    runs = []
    for run in header["runs"]:
        run_id = str(run["id"])
        problem = preview["problems"].get(run_id)
        already_here = kind.run_exists(data_dir, run_id)
        runs.append({
            "id": run_id,
            "name": str(run.get("name") or run_id),
            "workflow": str(run.get("workflow") or ""),
            "status": str(run.get("status") or ""),
            "created_at": str(run.get("created_at") or ""),
            "size_label": size_label(run["size_bytes"]) if isinstance(run.get("size_bytes"), int) else "",
            "problem": problem,
            "already_here": already_here,
            "cannot_resume": run_id in preview.get("cannot_resume", []),
            "checked": problem is None and not already_here,
        })
    source_version = str(header.get("grax_version") or "")
    return {
        "token": token,
        "kind": kind.key,
        "kind_label": kind.label,
        "name": str(header["object"].get("name") or header["object"]["id"]),
        "grax_version": source_version,
        "version_differs": source_version != _grax_version(),
        "local_version": _grax_version(),
        "exported_at": str(header.get("exported_at") or ""),
        "runs": runs,
        "matches": _matching_objects(kind, data_dir, header),
    }


def cancel_import(data_dir: Path, token: str) -> None:
    """Discard a staged package."""
    _discard_stage(_stage_dir(data_dir, token))


def _free_id(base: str, is_taken: Callable[[str], bool]) -> str:
    """Return ``base``, or ``base`` with the first free number."""
    candidate = base
    number = 2
    while is_taken(candidate):
        candidate = f"{base}-{number}"
        number += 1
    return candidate


def _copy_name(kind: ObjectKind, data_dir: Path, name: str) -> str:
    """Return a name for an added copy that tells it apart from local objects."""
    taken = {kind.object_name(spec).strip().casefold() for spec in kind.list_objects(data_dir)}
    candidate = f"{name} (imported)"
    number = 2
    while candidate.casefold() in taken:
        candidate = f"{name} (imported {number})"
        number += 1
    return candidate


def _with_profile_moved(
    manifest: dict[str, Any],
    profile: Path,
    kept: Path,
    resolve: Callable[[str], Path],
) -> dict[str, Any] | None:
    """Return a manifest that reads ``kept`` instead of ``profile``, or None if it never did."""
    reads_profile = False

    def repoint(value: str) -> str:
        nonlocal reads_profile
        if not value or resolve(value) != profile:
            return value
        reads_profile = True
        return str(kept)

    updated = _rewrite(manifest, repoint, lambda text: text)
    return updated if reads_profile else None


def _keep_profile_for_local_runs(kind: ObjectKind, data_dir: Path, profile: Path) -> None:
    """Give runs that read a profile about to be overwritten their own copy of it."""
    if not profile.is_file():
        return
    resolved = profile.resolve()
    store = kind.run_store(data_dir)
    kept_name = f"{sha1(str(resolved).encode('utf-8')).hexdigest()[:16]}.csv"
    # Sweep records repeat the same few paths thousands of times, and each lookup asks the disk.
    resolve = lru_cache(maxsize=None)(lambda value: Path(value).resolve())
    for manifest in store.list():
        run_id = str(manifest["id"])
        kept = store.run_dir(run_id) / "afm_profiles" / kept_name
        updated = _with_profile_moved(manifest, resolved, kept, resolve)
        if updated is not None:
            kept.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(profile, kept)
            store.update(run_id, lambda saved, change=updated: saved.update(change))


def _replace_file_with_undo(source: Path, target: Path, undo: list[Callable[[], None]], backups: Path) -> None:
    """Copy ``source`` over ``target``, keeping the old file until the import succeeds."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        backups.mkdir(parents=True, exist_ok=True)
        backup = backups / f"{uuid4().hex}-{target.name}"
        shutil.copy2(target, backup)
        undo.append(lambda: replace_file(backup, target))
    else:
        undo.append(lambda: target.unlink(missing_ok=True))
    temporary = target.with_name(f"{target.name}.{uuid4().hex}.tmp")
    shutil.copy2(source, temporary)
    replace_file(temporary, target)


def _failed_install_error(stage: Path, data_dir: Path, error: Exception) -> TransferError:
    """Restore the upload so the import can be tried again, and describe the failure."""
    message = f"The import failed and nothing was changed: {error}"
    try:
        remove_tree(stage / "extract")
        _unpack_package(stage, data_dir)
    except Exception:
        _discard_stage(stage)
        return TransferError(f"{message} Choose the file again.", can_retry=False)
    return TransferError(message)


def install_package(
    app: Any,
    data_dir: Path,
    token: str,
    *,
    run_ids: list[str],
    mode: str,
    target_id: str = "",
) -> dict[str, Any]:
    """Add a staged package's object and selected runs to the workspace, all or nothing."""
    with locked(_imports_dir(data_dir) / "install"):
        preview, stage = _load_stage(data_dir, token)
        header = preview["header"]
        kind = KINDS[header["kind"]]
        folder = stage / "extract"
        selected = list(dict.fromkeys(run_ids))
        for run_id in selected:
            if run_id not in preview["problems"]:
                raise TransferError(f"Run {run_id} is not in this file.")
            if preview["problems"][run_id]:
                raise TransferError(f"Run {run_id} cannot be imported. {preview['problems'][run_id]}")
        matches = _matching_objects(kind, data_dir, header)
        name = str(header["object"].get("name") or header["object"]["id"])
        target: dict[str, Any] | None = None
        if mode == "replace":
            if target_id not in {match["id"] for match in matches}:
                raise TransferError("Choose which object to replace.")
            target = kind.load(data_dir, target_id)
            if any(kind.is_run_active(app, run) for run in kind.linked_runs(data_dir, target_id)):
                raise TransferError("Stop this object's running runs before replacing it.")
        elif mode != "copy":
            raise TransferError("Choose whether to add a copy or replace the object.")
        original_id = str(header["object"]["id"])
        if target is not None:
            object_id = str(target["id"])
        else:
            if matches:
                name = _copy_name(kind, data_dir, name)
            object_id = _free_id(original_id, lambda value: kind.object_exists(data_dir, value))

        undo: list[Callable[[], None]] = []
        try:
            for run_id in selected:
                final_id = _free_id(run_id, lambda value: kind.run_exists(data_dir, value))
                final_dir = kind.run_dir(data_dir, final_id)
                staged = folder / "runs" / run_id
                _rewrite_run_files(staged, final_dir)
                manifest = read_json(staged / kind.manifest_name)
                manifest["id"] = final_id
                kind.relink_run(manifest, object_id=object_id, object_name=name)
                write_json_atomically(staged / kind.manifest_name, manifest, sort_keys=True)
                move_tree(staged, final_dir)
                undo.append(lambda path=final_dir: remove_tree(path))

            sidecar = kind.sidecar_folder(data_dir, object_id)
            object_files = [
                path for path in sorted((folder / "object").iterdir())
                if path.name != OBJECT_FILE and path.is_file()
            ]
            if target is None and object_files and not sidecar.exists():
                undo.append(lambda: remove_tree(sidecar))
            if target is not None and object_files:
                _keep_profile_for_local_runs(kind, data_dir, sidecar / "profile.csv")
            for path in object_files:
                _replace_file_with_undo(path, sidecar / path.name, undo, stage / "backups")

            spec = _restored_object(folder, sidecar)
            kind.set_object_name(spec, name)
            spec["id"] = object_id
            now = datetime.now().isoformat(timespec="seconds")
            if target is None:
                spec.setdefault("created_at", now)
                undo.append(lambda: kind.remove_object(data_dir, object_id))
            else:
                previous = deepcopy(target)
                undo.append(lambda: kind.write_object(data_dir, previous))
                undo.append(kind.local_run_restorer(data_dir, object_id))
                spec = kind.replace_local_object(data_dir, target=target, imported=spec)
            kind.stamp_saved_object(spec, now)
            kind.write_object(data_dir, spec)
            # A record that breaks the lists would take whole pages down, so check them now.
            kind.list_objects(data_dir)
            kind.linked_runs(data_dir, object_id)
        except Exception as error:
            for step in reversed(undo):
                try:
                    step()
                except Exception:
                    app.logger.exception("Could not undo part of a failed import.")
            app.logger.exception("An import failed and was undone.")
            raise _failed_install_error(stage, data_dir, error) from error
        _discard_stage(stage)
    return {
        "kind": kind.key,
        "kind_label": kind.label,
        "manage_endpoint": kind.manage_endpoint,
        "name": kind.object_name(spec),
        "id": object_id,
        "run_count": len(selected),
        "replaced": target is not None,
    }
