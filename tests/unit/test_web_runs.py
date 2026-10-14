"""Tests for saved web run manifests."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

from grax.web.runs import RunStore


@pytest.mark.skipif(os.name != "nt", reason="Windows file sharing semantics")
def test_save_retries_while_manifest_is_open_for_reading(tmp_path: Path) -> None:
    """A status reader must not make a progress update fail on Windows."""
    store = RunStore(tmp_path / "runs")
    store.save({"id": "study", "status": "queued", "completed_points": 0})
    manifest_path = tmp_path / "runs" / "study" / "manifest.json"
    errors: list[Exception] = []

    def save_progress() -> None:
        try:
            store.save({"id": "study", "status": "running", "completed_points": 1})
        except Exception as error:
            errors.append(error)

    with manifest_path.open("r", encoding="utf-8") as reader:
        worker = threading.Thread(target=save_progress)
        worker.start()
        deadline = time.monotonic() + 1.0
        while (
            not list(manifest_path.parent.glob("manifest.json.*.tmp"))
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert worker.is_alive()
        assert '"completed_points": 0' in reader.read()

    worker.join(timeout=3.0)
    assert not worker.is_alive()
    assert not errors
    assert store.load("study")["completed_points"] == 1
    assert not list(manifest_path.parent.glob("manifest.json.*.tmp"))
