"""File writes and deletes that background jobs and requests can share safely.

Jobs and page requests touch the same run folders from different threads, and on
Windows a file that a browser or another thread is reading cannot be replaced or
deleted for a moment. Everything here retries through those brief locks and
serializes read-modify-write updates of one file.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

_RETRY_SECONDS = 2.0
_RETRY_DELAY = 0.02

_path_locks: dict[str, threading.RLock] = {}
_path_locks_guard = threading.Lock()


class FileInUseError(PermissionError):
    """Raised when another program keeps a file open, so it cannot be deleted yet."""

    def __init__(self, path: Path, *, folder: bool = False) -> None:
        """Name the open file, or for ``folder`` the folder that holds it."""

        what = f"A file in {Path(path).name} is" if folder else f"{Path(path).name} is"
        super().__init__(
            f"{what} open in another program, so it could not be deleted. Close it and try again."
        )


@contextmanager
def locked(path: Path) -> Iterator[None]:
    """Hold this process's lock for one file while it is read, changed and written."""

    key = os.path.normcase(str(Path(path).resolve()))
    with _path_locks_guard:
        lock = _path_locks.setdefault(key, threading.RLock())
    with lock:
        yield


def _retry_windows_lock(action: Callable[[], Any]) -> Any:
    """Run ``action``, retrying briefly while Windows reports the file as in use."""

    deadline = time.monotonic() + _RETRY_SECONDS
    while True:
        try:
            return action()
        except PermissionError:
            if os.name != "nt" or time.monotonic() >= deadline:
                raise
            time.sleep(_RETRY_DELAY)


def replace_file(source: Path, destination: Path) -> None:
    """Move ``source`` over ``destination`` in one step.

    On Windows a program that keeps ``destination`` open, such as an image
    viewer showing a run's plot, blocks the move for as long as it has the file
    open. Past the brief retry the new content is copied into the open file
    instead, so the save still lands; only then can a reader see it change.
    """

    try:
        _retry_windows_lock(lambda: os.replace(source, destination))
    except PermissionError:
        if os.name != "nt" or not Path(destination).is_file():
            raise
        with open(source, "rb") as new, open(destination, "r+b") as old:
            shutil.copyfileobj(new, old)
            old.truncate()
        Path(source).unlink(missing_ok=True)


def read_text(path: Path) -> str:
    """Read a UTF-8 file, waiting out the moment Windows holds it during a replace."""

    return _retry_windows_lock(lambda: Path(path).read_text(encoding="utf-8"))


def read_json(path: Path) -> Any:
    """Read and parse a JSON file that a job may be replacing right now."""

    return json.loads(read_text(path))


def write_text_atomically(path: Path, text: str) -> None:
    """Write ``text`` so readers see either the old or the new file, never a partial one."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        replace_file(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomically(path: Path, payload: Any, *, sort_keys: bool = False) -> None:
    """Write ``payload`` as indented JSON with a trailing newline, atomically."""

    write_text_atomically(path, json.dumps(payload, indent=2, sort_keys=sort_keys) + "\n")


def update_json(path: Path, change: Callable[[Any], Any], *, sort_keys: bool = False) -> Any:
    """Load a JSON file, apply ``change`` and save it, with no other update in between.

    ``change`` edits the loaded value in place or returns a replacement.
    """

    with locked(path):
        payload = read_json(path)
        result = change(payload)
        if result is not None:
            payload = result
        write_json_atomically(path, payload, sort_keys=sort_keys)
        return payload


def _clear_read_only_and_retry(
    function: Callable[[str], Any], failed: str, error: BaseException
) -> None:
    if isinstance(error, FileNotFoundError):
        return
    if not isinstance(error, PermissionError):
        raise error
    try:
        os.chmod(failed, stat.S_IWRITE)
    except OSError:
        pass
    _retry_windows_lock(lambda: function(failed))


def remove_file(path: Path) -> None:
    """Delete one file, clearing a Windows read-only flag and waiting out brief file locks."""

    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass
    except PermissionError as error:
        try:
            _clear_read_only_and_retry(os.unlink, str(path), error)
        except PermissionError as still_open:
            raise FileInUseError(path) from still_open


def remove_tree(path: Path, *, missing_ok: bool = True) -> None:
    """Delete a folder, clearing Windows read-only flags and waiting out brief file locks.

    The folder is first moved aside in one step. Windows refuses that move while
    another program keeps a file inside open, so such a folder is left whole and
    ``FileInUseError`` is raised, instead of losing some files (perhaps the
    manifest that lists it) and keeping others.
    """

    path = Path(path)
    if missing_ok and not path.exists():
        return
    # One level down, so the "*/manifest.json" scans never see a folder mid-delete.
    trash = path.parent / f".deleting-{uuid4().hex}" / path.name
    trash.parent.mkdir()
    try:
        try:
            _retry_windows_lock(lambda: os.rename(path, trash))
        except PermissionError as error:
            raise FileInUseError(path, folder=True) from error
        shutil.rmtree(trash, onexc=_clear_read_only_and_retry)
    finally:
        try:
            trash.parent.rmdir()
        except OSError:
            pass
