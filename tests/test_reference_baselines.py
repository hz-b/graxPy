from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_reference_baselines_match_saved_outputs() -> None:
    """Run fast reference examples and compare them with saved baseline CSVs."""

    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, "compare_reference.py"],
        cwd=repo_root / "examples" / "reference_baselines",
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
