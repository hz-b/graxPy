from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("ax", reason="Ax is only installed with the 'opt' extra.")

from grax_opt.checkpoint import (  # noqa: E402
    ax_trial_count,
    load_ax_client_snapshot,
    save_ax_client_snapshot,
)
from grax_opt.optimize import _import_ax_client, _import_objective_properties  # noqa: E402


def _build_ax_client():
    ax_client_cls = _import_ax_client()
    objective_properties = _import_objective_properties()
    ax_client = ax_client_cls(random_seed=7)
    ax_client.create_experiment(
        name="snapshot_smoke",
        parameters=[
            {"name": "x", "type": "range", "bounds": [0.0, 1.0], "value_type": "float"}
        ],
        objectives={"loss": objective_properties(minimize=True)},
    )
    return ax_client


def _run_trials(ax_client, count: int) -> None:
    for _ in range(count):
        parameters, trial_index = ax_client.get_next_trial()
        loss = float(parameters["x"]) ** 2
        ax_client.complete_trial(trial_index=trial_index, raw_data={"loss": (loss, 1.0e-6)})


def test_ax_client_snapshot_round_trip_continues_the_experiment(tmp_path: Path) -> None:
    ax_client = _build_ax_client()
    _run_trials(ax_client, 3)
    snapshot_path = tmp_path / "ax_client_snapshot.json"

    save_ax_client_snapshot(ax_client, snapshot_path)
    restored = load_ax_client_snapshot(snapshot_path)

    assert ax_trial_count(restored) == 3

    _run_trials(restored, 2)
    assert ax_trial_count(restored) == 5

    _parameters, trial_index = restored.get_next_trial()
    assert trial_index == 5


def test_save_ax_client_snapshot_leaves_no_temporary_file(tmp_path: Path) -> None:
    ax_client = _build_ax_client()
    _run_trials(ax_client, 1)
    snapshot_path = tmp_path / "ax_client_snapshot.json"

    save_ax_client_snapshot(ax_client, snapshot_path)

    assert snapshot_path.is_file()
    assert [path.name for path in tmp_path.iterdir() if ".tmp" in path.name] == []
