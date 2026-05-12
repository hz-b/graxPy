from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Mark tests that need checkpoint cleanup."""
    config.addinivalue_line(
        "markers", "no_checkpoint: skip checkpoint file checks for this test"
    )


@pytest.fixture(autouse=True)
def _cleanup_checkpoints() -> None:
    """Clean up checkpoint directory before and after each test."""
    import shutil
    from pathlib import Path

    checkpoint_dir = Path("results/checkpoints")
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    yield
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
