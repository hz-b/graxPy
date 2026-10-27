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
    """Move ``source`` over ``destination`` in one step."""

    _retry_windows_lock(lambda: os.replace(source, destination))


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
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        result = change(payload)
        if result is not None:
            payload = result
        write_json_atomically(path, payload, sort_keys=sort_keys)
        return payload


def remove_tree(path: Path, *, missing_ok: bool = True) -> None:
    """Delete a folder, clearing Windows read-only flags and waiting out brief file locks."""

    path = Path(path)
    if missing_ok and not path.exists():
        return

    def retry(function: Callable[[str], Any], failed: str, error: BaseException) -> None:
        if isinstance(error, FileNotFoundError):
            return
        if not isinstance(error, PermissionError):
            raise error
        try:
            os.chmod(failed, stat.S_IWRITE)
        except OSError:
            pass
        _retry_windows_lock(lambda: function(failed))

    shutil.rmtree(path, onexc=retry)
