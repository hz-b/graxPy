"""File-based run persistence and comparison helpers for the local web app."""

from __future__ import annotations

import csv
import json
import re
import shutil
from datetime import datetime
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class RunOrderSelection:
    """Selected diffraction orders for one saved run."""

    run_id: str
    orders: tuple[int, ...]


class RunStore:
    """Store run manifests and derived artifacts in a filesystem directory."""

    def __init__(self, directory: str | Path) -> None:
        """Initialize the run store.

        Args:
            directory: Directory containing run subdirectories.
        """
        self.directory = Path(directory)

    def list(self) -> list[dict[str, Any]]:
        """Return saved run manifests ordered from newest to oldest."""
        if not self.directory.exists():
            return []
        runs = [self.load(path.parent.name) for path in self.directory.glob("*/manifest.json")]
        return sorted(
            runs,
            key=lambda run: (
                str(run.get("created_at", "")),
                str(run.get("updated_at", "")),
                str(run.get("id", "")),
            ),
            reverse=True,
        )

    def load(self, run_id: str) -> dict[str, Any]:
        """Load one run manifest by identifier."""
        path = self._manifest_path(run_id)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload.setdefault("id", run_id)
        payload.setdefault("display_name", self._default_display_name(payload))
        return payload

    def save(self, manifest: dict[str, Any]) -> dict[str, Any]:
        """Persist a run manifest and return the saved payload."""
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = dict(manifest)
        run_id = str(payload["id"])
        payload["id"] = run_id
        payload.setdefault("display_name", self._default_display_name(payload))
        path = self._manifest_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f"{path.name}.{datetime.now().timestamp():.9f}.tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temp_path.replace(path)
        return payload

    def rename(self, run_id: str, display_name: str) -> dict[str, Any]:
        """Update the user-visible display name for one run."""
        payload = self.load(run_id)
        name = display_name.strip()
        if name:
            payload["display_name"] = name
        else:
            payload.pop("display_name", None)
        payload["updated_at"] = _timestamp()
        return self.save(payload)

    def update_comment(self, run_id: str, comment: str) -> dict[str, Any]:
        """Update the saved user comment for one run."""

        payload = self.load(run_id)
        payload["comment"] = comment.strip()
        payload["updated_at"] = _timestamp()
        return self.save(payload)

    def delete_many(self, run_ids: Sequence[str]) -> None:
        """Delete several run directories."""
        for run_id in run_ids:
            run_dir = self._run_dir(run_id)
            if run_dir.exists():
                shutil.rmtree(run_dir)

    def available_orders(self, run_id: str) -> list[int]:
        """Return the diffraction orders stored for one run."""
        rows = self._all_orders_rows(run_id)
        orders = {int(row["order"]) for row in rows}
        return sorted(orders)

    def order_series(
        self,
        run_id: str,
        diffraction_order: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return energy and efficiency arrays for one order in one run."""
        rows = self._all_orders_rows(run_id)
        filtered = [
            row
            for row in rows
            if int(row["order"]) == int(diffraction_order)
        ]
        filtered.sort(key=lambda row: float(row["energy_ev"]))
        energies = np.asarray([float(row["energy_ev"]) for row in filtered], dtype=float)
        efficiencies = np.asarray([float(row["efficiency"]) for row in filtered], dtype=float)
        return energies, efficiencies

    def _all_orders_rows(self, run_id: str) -> list[dict[str, str]]:
        """Read the saved all-orders CSV for one run."""
        path = self._run_dir(run_id) / "all_orders.csv"
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def _run_dir(self, run_id: str) -> Path:
        """Return the filesystem path for one run identifier."""
        safe_id = _slugify(run_id)
        if safe_id != run_id:
            raise ValueError("Invalid run id.")
        return self.directory / run_id

    def _manifest_path(self, run_id: str) -> Path:
        """Return the manifest path for one run identifier."""
        return self._run_dir(run_id) / "manifest.json"

    def _default_display_name(self, manifest: dict[str, Any]) -> str:
        """Return a fallback display name for a run."""
        grating_name = str(manifest.get("grating_name", "Run"))
        workflow = str(manifest.get("workflow", "run"))
        return f"{grating_name} · {workflow}"


def build_run_comparison_figure(
    selections: Sequence[RunOrderSelection],
    *,
    store: RunStore,
    output_filename: str | Path,
) -> list[str]:
    """Save a comparison plot for multiple saved runs and return the plotted labels."""
    figure, axis = plt.subplots(figsize=(10, 6))
    markers = ["o", "s", "^", "d", "v", "x"]
    labels: list[str] = []
    for index, selection in enumerate(selections):
        manifest = store.load(selection.run_id)
        run_name = str(manifest.get("display_name") or manifest.get("name") or selection.run_id)
        for order in selection.orders:
            energies, efficiencies = store.order_series(selection.run_id, order)
            if energies.size == 0:
                continue
            label = f"{run_name} · order {order}"
            labels.append(label)
            axis.plot(
                energies,
                efficiencies,
                f"{markers[index % len(markers)]}-",
                linewidth=1.0,
                markersize=3.0,
                label=label,
            )
    axis.set_xlabel("Photon Energy (eV)")
    axis.set_ylabel("Diffraction Efficiency")
    axis.set_title("Saved run comparison")
    axis.grid(True, alpha=0.3)
    if labels:
        axis.legend(loc="best")
    figure.tight_layout()
    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_filename, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return labels


def _slugify(value: str) -> str:
    """Return a filesystem-safe identifier."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return re.sub(r"-{2,}", "-", slug)


def _timestamp() -> str:
    """Return an ISO-like timestamp for manifest updates."""
    from datetime import datetime

    return datetime.now().isoformat(timespec="seconds")
