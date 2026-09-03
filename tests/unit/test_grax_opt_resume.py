from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from grax_opt import checkpoint as checkpoint_module
from grax_opt import dynamic as dynamic_module
from grax_opt import joint as joint_module
from grax_opt import objective as objective_module
from grax_opt.checkpoint import OptimizerCheckpointPaths
from grax_opt.joint import optimize_to_joint_measurements
from grax_opt.loop import is_significant_improvement


class _FakeRunner:
    resolved_max_workers = 1

    call_count = 0
    interrupt_at: int | None = None

    def __init__(self, **_kwargs: object) -> None:
        pass

    def run_cases(self, cases, metadata=None):
        type(self).call_count += 1
        if type(self).interrupt_at is not None and type(self).call_count == type(self).interrupt_at:
            raise KeyboardInterrupt("interrupted mid-run")
        for index, case in enumerate(cases):
            yield SimpleNamespace(
                index=index,
                case_id=case["case_id"],
                status="ok",
                selected_efficiency=0.3,
            )


class _FakeAxClient:
    def __init__(self, start_index: int = 0) -> None:
        self.next_index = start_index
        self.completed: list[int] = []

    def create_experiment(self, **_kwargs: object) -> None:
        return None

    def get_next_trial(self):
        trial_index = self.next_index
        self.next_index += 1
        return {"depth_nm": 5.0 + trial_index}, trial_index

    def complete_trial(self, trial_index, raw_data=None, data=None) -> None:
        self.completed.append(int(trial_index))

    def save_to_json_file(self, filepath: str) -> None:
        Path(filepath).write_text(json.dumps({"next_index": self.next_index}), encoding="utf-8")

    @property
    def experiment(self) -> SimpleNamespace:
        return SimpleNamespace(trials={index: None for index in range(self.next_index)})


def _install_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeRunner.call_count = 0
    _FakeRunner.interrupt_at = None
    monkeypatch.setattr(objective_module, "BatchSimulationRunner", _FakeRunner)
    monkeypatch.setattr(
        dynamic_module,
        "_create_ax_client_for_measurement_fit_config",
        lambda _config: _FakeAxClient(),
    )
    monkeypatch.setattr(
        joint_module,
        "_create_ax_client_for_joint_config",
        lambda _config: _FakeAxClient(),
    )
    monkeypatch.setattr(
        checkpoint_module,
        "load_ax_client_snapshot",
        lambda snapshot_path, recorded_ax_version=None: _FakeAxClient(
            start_index=int(json.loads(Path(snapshot_path).read_text(encoding="utf-8"))["next_index"])
        ),
    )


def _write_measurement(path: Path, rows: list[tuple[float, float]]) -> Path:
    path.write_text(
        "".join(f"{energy_ev} {efficiency}\n" for energy_ev, efficiency in rows),
        encoding="utf-8",
    )
    return path


def _spec(tmp_path: Path, **overrides: object) -> dict[str, object]:
    measurement_path = tmp_path / "m.dat"
    if not measurement_path.is_file():
        _write_measurement(measurement_path, [(100.0, 0.2), (200.0, 0.3), (300.0, 0.4)])
    spec: dict[str, object] = {
        "build_grating": lambda _parameters: SimpleNamespace(period_lpermm=2000.0),
        "parameter_bounds": {"depth_nm": (1.0, 20.0)},
        "measurement_path": measurement_path,
        "output_dir": tmp_path / "out",
        "evaluation_energies_ev": [100.0, 200.0, 300.0],
        "total_trials": 5,
        "save_best_fit_plot": False,
        "save_loss_plot": False,
    }
    spec.update(overrides)
    return spec


def test_is_significant_improvement_requires_minimum_relative_gain() -> None:
    assert is_significant_improvement(float("inf"), 1.0, 0.5) is True
    assert is_significant_improvement(1.0, 0.4, 0.5) is True
    assert is_significant_improvement(1.0, 0.9, 0.5) is False
    assert is_significant_improvement(1.0, 2.0, 0.5) is False
    assert is_significant_improvement(1.0, 0.999, 0.0) is True


def test_resume_defaults_checkpoint_dir_to_output_dir(tmp_path: Path) -> None:
    paths = OptimizerCheckpointPaths.for_config(
        SimpleNamespace(output_dir=tmp_path / "out", checkpoint_dir=None)
    )

    assert paths.checkpoint_dir == tmp_path / "out" / "checkpoint"
    assert paths.ax_snapshot_path.name == "ax_client_snapshot.json"
    assert paths.state_path.name == "optimizer_state.json"
    assert paths.trial_records_path.name == "trial_records.jsonl"


def test_run_writes_checkpoint_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fakes(monkeypatch)

    dynamic_module.optimize_to_measurements(_spec(tmp_path))

    checkpoint_dir = tmp_path / "out" / "checkpoint"
    assert sorted(path.name for path in checkpoint_dir.iterdir()) == [
        "ax_client_snapshot.json",
        "optimizer_state.json",
        "trial_records.jsonl",
    ]


def test_fresh_run_truncates_a_previous_trial_record_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)
    dynamic_module.optimize_to_measurements(_spec(tmp_path, total_trials=5))
    trial_records_path = tmp_path / "out" / "checkpoint" / "trial_records.jsonl"
    assert len(trial_records_path.read_text(encoding="utf-8").strip().splitlines()) == 5

    dynamic_module.optimize_to_measurements(_spec(tmp_path, total_trials=3))

    records = [
        json.loads(line)
        for line in trial_records_path.read_text(encoding="utf-8").strip().splitlines()
    ]
    assert len(records) == 3
    assert [record["trial_index"] for record in records] == [0, 1, 2]


def test_resume_after_a_fresh_rerun_does_not_double_count_trials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)
    dynamic_module.optimize_to_measurements(_spec(tmp_path, total_trials=5))
    dynamic_module.optimize_to_measurements(_spec(tmp_path, total_trials=3))

    result = dynamic_module.optimize_to_measurements(_spec(tmp_path, total_trials=6, resume=True))

    assert result.completed_trials == 6
    rows = list(csv.reader(result.trial_history_csv_path.open(encoding="utf-8")))
    assert [row[0] for row in rows[1:]] == [str(index) for index in range(6)]


def test_resume_with_missing_checkpoint_starts_fresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)

    result = dynamic_module.optimize_to_measurements(_spec(tmp_path, resume=True))

    assert result.completed_trials == 5


def test_resume_restores_trials_and_extends_total_trials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)

    first = dynamic_module.optimize_to_measurements(_spec(tmp_path, total_trials=5))
    calls_after_first_run = _FakeRunner.call_count
    assert first.completed_trials == 5
    assert calls_after_first_run == 5

    second = dynamic_module.optimize_to_measurements(
        _spec(tmp_path, total_trials=12, resume=True)
    )

    assert second.completed_trials == 12
    assert _FakeRunner.call_count - calls_after_first_run == 7

    rows = list(csv.reader(second.trial_history_csv_path.open(encoding="utf-8")))
    assert len(rows) - 1 == 12
    assert [row[0] for row in rows[1:]] == [str(index) for index in range(12)]


def test_resume_does_not_rerun_when_total_trials_already_reached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)
    dynamic_module.optimize_to_measurements(_spec(tmp_path, total_trials=5))
    calls_after_first_run = _FakeRunner.call_count

    result = dynamic_module.optimize_to_measurements(_spec(tmp_path, total_trials=5, resume=True))

    assert _FakeRunner.call_count == calls_after_first_run
    assert result.completed_trials == 5
    assert result.result_json_path.is_file()


def test_resume_rejects_changed_parameter_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)
    dynamic_module.optimize_to_measurements(_spec(tmp_path))

    with pytest.raises(ValueError, match="different optimization problem"):
        dynamic_module.optimize_to_measurements(
            _spec(
                tmp_path,
                total_trials=9,
                resume=True,
                parameter_bounds={"depth_nm": (1.0, 99.0)},
            )
        )


def test_resume_rejects_changed_measurement_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)
    dynamic_module.optimize_to_measurements(_spec(tmp_path))
    _write_measurement(tmp_path / "m.dat", [(100.0, 0.9), (200.0, 0.8), (300.0, 0.7)])

    with pytest.raises(ValueError, match="different optimization problem"):
        dynamic_module.optimize_to_measurements(_spec(tmp_path, total_trials=9, resume=True))


def test_resume_ignores_torn_trial_record_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)
    dynamic_module.optimize_to_measurements(_spec(tmp_path, total_trials=5))
    trial_records_path = tmp_path / "out" / "checkpoint" / "trial_records.jsonl"
    trial_records_path.write_text(
        trial_records_path.read_text(encoding="utf-8") + '{"trial_index": 9, "loss": ',
        encoding="utf-8",
    )

    result = dynamic_module.optimize_to_measurements(_spec(tmp_path, total_trials=6, resume=True))

    assert result.completed_trials == 6


def test_resume_raises_on_corrupt_ax_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)
    dynamic_module.optimize_to_measurements(_spec(tmp_path))
    monkeypatch.setattr(
        checkpoint_module,
        "load_ax_client_snapshot",
        lambda snapshot_path, recorded_ax_version=None: (_ for _ in ()).throw(
            ValueError("Cannot resume: the Ax snapshot could not be loaded.")
        ),
    )

    with pytest.raises(ValueError, match="Cannot resume"):
        dynamic_module.optimize_to_measurements(_spec(tmp_path, total_trials=9, resume=True))


def test_resume_raises_on_incomplete_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)
    dynamic_module.optimize_to_measurements(_spec(tmp_path))
    (tmp_path / "out" / "checkpoint" / "optimizer_state.json").unlink()

    with pytest.raises(ValueError, match="incomplete"):
        dynamic_module.optimize_to_measurements(_spec(tmp_path, total_trials=9, resume=True))


def test_checkpoint_writes_leave_no_temporary_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)

    dynamic_module.optimize_to_measurements(_spec(tmp_path))

    checkpoint_dir = tmp_path / "out" / "checkpoint"
    assert [path.name for path in checkpoint_dir.iterdir() if ".tmp" in path.name] == []
    assert [path.name for path in (tmp_path / "out").iterdir() if ".tmp" in path.name] == []


def test_resume_preserves_best_across_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)
    first = dynamic_module.optimize_to_measurements(_spec(tmp_path, total_trials=3))

    second = dynamic_module.optimize_to_measurements(_spec(tmp_path, total_trials=6, resume=True))

    assert second.best_loss == pytest.approx(first.best_loss)
    assert second.best_parameters == first.best_parameters


def test_resume_accumulates_elapsed_seconds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)
    dynamic_module.optimize_to_measurements(_spec(tmp_path, total_trials=3))
    state_path = tmp_path / "out" / "checkpoint" / "optimizer_state.json"
    first_state = json.loads(state_path.read_text(encoding="utf-8"))

    dynamic_module.optimize_to_measurements(_spec(tmp_path, total_trials=6, resume=True))
    second_state = json.loads(state_path.read_text(encoding="utf-8"))

    assert second_state["run_count"] == first_state["run_count"] + 1
    assert second_state["created"] == first_state["created"]
    assert second_state["total_trials_history"] == [3, 6]
    assert (
        second_state["cumulative_elapsed_seconds"] >= first_state["cumulative_elapsed_seconds"]
    )


def test_created_timestamp_marks_the_run_start_not_the_last_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)

    dynamic_module.optimize_to_measurements(_spec(tmp_path, total_trials=4))

    state = json.loads(
        (tmp_path / "out" / "checkpoint" / "optimizer_state.json").read_text(encoding="utf-8")
    )
    assert state["created"] <= state["current_run_started"]
    assert state["created"] < state["last_updated"]


def test_optimize_to_joint_measurements_resumes_and_extends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)
    first_path = _write_measurement(tmp_path / "m1.dat", [(100.0, 0.2), (200.0, 0.3)])
    second_path = _write_measurement(tmp_path / "m2.dat", [(100.0, 0.4), (200.0, 0.5)])

    def joint_spec(total_trials: int, resume: bool) -> dict[str, object]:
        return {
            "build_grating": lambda _parameters: SimpleNamespace(period_lpermm=2000.0),
            "parameter_bounds": {"depth_nm": (1.0, 20.0)},
            "output_dir": tmp_path / "out",
            "measurements": [
                {"grazing_angle_deg": 1.0, "measurement_path": first_path},
                {"grazing_angle_deg": 2.0, "measurement_path": second_path},
            ],
            "total_trials": total_trials,
            "resume": resume,
            "save_best_fit_plot": False,
            "save_loss_plot": False,
        }

    optimize_to_joint_measurements(joint_spec(4, False))
    calls_after_first_run = _FakeRunner.call_count

    result = optimize_to_joint_measurements(joint_spec(10, True))

    assert result.completed_trials == 10
    assert _FakeRunner.call_count - calls_after_first_run == 6


def test_interrupted_run_with_checkpoint_interval_does_not_over_run_the_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resume must not replay trials the Ax snapshot never recorded.

    The trial-record log is appended every trial but the snapshot and run state
    are only rewritten every ``checkpoint_interval`` trials, so an interruption
    leaves the log ahead of the snapshot. Those extra records describe trials Ax
    will generate again; keeping them double-counted the trials and pushed the
    resumed run past ``total_trials``.
    """

    _install_fakes(monkeypatch)
    checkpoint_dir = tmp_path / "out" / "checkpoint"

    _FakeRunner.interrupt_at = 5
    with pytest.raises(KeyboardInterrupt):
        dynamic_module.optimize_to_measurements(
            _spec(tmp_path, total_trials=6, checkpoint_interval=3)
        )

    recorded_before_resume = [
        line for line in (checkpoint_dir / "trial_records.jsonl").read_text().splitlines() if line
    ]
    persisted_cursor = json.loads((checkpoint_dir / "optimizer_state.json").read_text())[
        "trial_index_cursor"
    ]
    assert len(recorded_before_resume) > persisted_cursor

    _FakeRunner.interrupt_at = None
    result = dynamic_module.optimize_to_measurements(
        _spec(tmp_path, total_trials=6, checkpoint_interval=3, resume=True)
    )

    assert result.completed_trials == 6
    trial_indices = [
        json.loads(line)["trial_index"]
        for line in (checkpoint_dir / "trial_records.jsonl").read_text().splitlines()
        if line
    ]
    assert trial_indices == list(range(6))


def test_measurement_fingerprint_refuses_an_unreadable_file(tmp_path: Path) -> None:
    """The content hash must not degrade to a shared sentinel.

    Both optimizer entry points load their measurements before building the
    fingerprint, so an unreadable file normally fails earlier with a load error
    and this guard is not reached. It matters if that ordering ever changes: a
    sentinel digest makes two different unreadable files compare equal, which
    would let a resume pass its data-integrity check against measurements that
    are no longer on disk.
    """

    with pytest.raises(ValueError, match="Cannot fingerprint the measurement file"):
        checkpoint_module._file_content_hash(tmp_path / "absent.dat")


def test_resume_is_refused_when_the_solver_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The solver is part of the problem, not a performance knob.

    ``total_trials`` and ``max_workers`` may change between runs, but swapping
    the electromagnetic solver changes the physics the surrogate model was fitted
    against, so it must block the resume rather than silently continue.
    """

    _install_fakes(monkeypatch)
    dynamic_module.optimize_to_measurements(_spec(tmp_path, total_trials=3))

    with pytest.raises(ValueError, match="changed: .*solver"):
        dynamic_module.optimize_to_measurements(
            _spec(tmp_path, total_trials=6, solver="neviere", resume=True)
        )


def test_resume_is_refused_when_a_measurement_condition_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch)
    first_path = _write_measurement(tmp_path / "m1.dat", [(100.0, 0.2), (200.0, 0.3)])
    second_path = _write_measurement(tmp_path / "m2.dat", [(100.0, 0.4), (200.0, 0.5)])

    def joint_spec(polarization: str, resume: bool) -> dict[str, object]:
        return {
            "build_grating": lambda _parameters: SimpleNamespace(period_lpermm=2000.0),
            "parameter_bounds": {"depth_nm": (1.0, 20.0)},
            "output_dir": tmp_path / "out",
            "measurements": [
                {"grazing_angle_deg": 1.0, "measurement_path": first_path},
                {
                    "grazing_angle_deg": 2.0,
                    "measurement_path": second_path,
                    "polarization": polarization,
                },
            ],
            "total_trials": 3,
            "resume": resume,
            "save_best_fit_plot": False,
            "save_loss_plot": False,
        }

    optimize_to_joint_measurements(joint_spec("s", False))

    with pytest.raises(ValueError, match="changed: measurements"):
        optimize_to_joint_measurements(joint_spec("p", True))
