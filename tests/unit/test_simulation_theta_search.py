from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from grax import simulation as simulation_module
from grax.simulation import (
    BatchSimulationRunner,
    SingleSimulationResult,
    multilayer_theta_search_cases,
    run_multilayer_theta_search_sweep,
)
from grax.simulation import batch as simulation_batch_module
from tests.simulation_helpers import (
    build_blazed_multilayer_angle_parity_grating,
    build_test_grating,
    fake_single_result,
)


def test_multilayer_theta_search_sweep_accumulates_elapsed_time_across_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run_multilayer_theta_search(**kwargs: object) -> SingleSimulationResult:
        energy_ev = float(kwargs["energy_ev"])
        diagnostics = simulation_module.ThetaSearchDiagnostics(
            estimated_grazing_angle_deg=1.2,
            rough_grazing_angles_deg=np.asarray([1.1, 1.2, 1.3], dtype=float),
            rough_efficiencies=np.asarray([0.2, 0.3, 0.25], dtype=float),
            precise_grazing_angles_deg=np.asarray([1.18, 1.2, 1.22], dtype=float),
            precise_efficiencies=np.asarray([0.31, 0.35, 0.33], dtype=float),
            selected_grazing_angle_deg=1.2,
            selected_efficiency=0.35,
            precise_fwhm_deg=0.04,
        )
        return SingleSimulationResult(
            energy_ev=energy_ev,
            grazing_angle_deg=1.2,
            orders=np.asarray([-1, 0, 1], dtype=int),
            selected_efficiency=0.35,
            selected_diffraction_angle_deg=2.0,
            efficiency_all=np.asarray([0.35, 0.0, 0.0], dtype=float),
            diffraction_angle_all=np.asarray([2.0, 1.0, 0.0], dtype=float),
            diffraction_order=int(kwargs["diffraction_order"]),
            fourier_orders=int(kwargs["final_fourier_orders"]),
            theta_search_diagnostics=diagnostics,
        )

    monkeypatch.setattr(simulation_module, "run_multilayer_theta_search", fake_run_multilayer_theta_search)

    first = run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0],
        output_dir=tmp_path,
        checkpoint_dir=tmp_path / "checkpoints",
        resume=False,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
    )
    metadata_path = tmp_path / "checkpoints" / "metadata.json"
    first_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    first_metadata["cumulative_elapsed_seconds"] = 12.0
    metadata_path.write_text(json.dumps(first_metadata, indent=2), encoding="utf-8")
    resumed = run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0, 1802.0],
        output_dir=tmp_path,
        checkpoint_dir=tmp_path / "checkpoints",
        resume=True,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert first.current_run_elapsed_seconds >= 0.0
    assert first.total_elapsed_seconds == pytest.approx(first.current_run_elapsed_seconds)
    assert resumed.current_run_elapsed_seconds >= 0.0
    assert resumed.total_elapsed_seconds >= 12.0
    assert metadata["last_run_elapsed_seconds"] == pytest.approx(resumed.current_run_elapsed_seconds)
    assert metadata["cumulative_elapsed_seconds"] == pytest.approx(resumed.total_elapsed_seconds)


def test_multilayer_theta_search_sweep_retries_on_threshold(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    call_count = {"value": 0}

    def fake_run_multilayer_theta_search(**kwargs: object) -> SingleSimulationResult:
        call_count["value"] += 1
        selected_efficiency = 5e-5 if call_count["value"] == 1 else 2e-4
        return fake_single_result(
            energy_ev=float(kwargs["energy_ev"]),
            grazing_angle_deg=1.2,
            selected_efficiency=selected_efficiency,
        )

    monkeypatch.setattr(simulation_module, "run_multilayer_theta_search", fake_run_multilayer_theta_search)
    sweep = run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0],
        output_dir=tmp_path,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
        retry_on_selected_efficiency_zero=True,
        retry_selected_efficiency_threshold=1e-4,
        max_zero_efficiency_retries=1,
    )

    assert call_count["value"] == 2
    assert len(sweep.batch_result.cases) == 1
    result = sweep.batch_result.cases[0]
    assert result.retry_triggered is True
    assert result.retry_attempts == 1
    assert result.retry_status == "recovered"
    assert result.selected_efficiency_below_retry_threshold is False


def test_batch_runner_theta_retry_uses_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[float] = []

    def fake_run_case_payload(
        payload: dict[str, object],
        *,
        diagnostic_callback: object = None,
    ) -> SingleSimulationResult:
        del diagnostic_callback
        calls.append(float(payload["energy_ev"]))
        selected_efficiency = 5e-5 if len(calls) == 1 else 2e-4
        return fake_single_result(
            energy_ev=float(payload["energy_ev"]),
            grazing_angle_deg=1.2,
            selected_efficiency=selected_efficiency,
        )

    monkeypatch.setattr(simulation_batch_module, "_run_case_payload", fake_run_case_payload)
    cases = list(
        multilayer_theta_search_cases(
            grating=build_blazed_multilayer_angle_parity_grating(),
            energies_ev=[2000.0],
        )
    )
    runner = BatchSimulationRunner(
        retry_on_selected_efficiency_zero=True,
        retry_selected_efficiency_threshold=1e-4,
        max_zero_efficiency_retries=1,
    )
    results = list(runner.run_cases(cases))

    assert len(calls) == 2
    assert len(results) == 1
    result = results[0]
    assert result.retry_triggered is True
    assert result.retry_attempts == 1
    assert result.retry_status == "recovered"
    assert result.selected_efficiency_below_retry_threshold is False


def test_retry_selected_efficiency_threshold_validation() -> None:
    with pytest.raises(ValueError, match="retry_selected_efficiency_threshold"):
        BatchSimulationRunner(retry_selected_efficiency_threshold=-1.0)
    with pytest.raises(ValueError, match="retry_selected_efficiency_threshold"):
        run_multilayer_theta_search_sweep(
            grating=build_blazed_multilayer_angle_parity_grating(),
            energies_ev=[1800.0],
            output_dir=Path("unused"),
            retry_selected_efficiency_threshold=-1.0,
        )


def test_batch_runner_removed_constructor_args_raise_type_error() -> None:
    with pytest.raises(TypeError, match="total_cases"):
        BatchSimulationRunner(total_cases=2)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="live_theta_scan_plot"):
        BatchSimulationRunner(live_theta_scan_plot=True)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="max_total_reflected_efficiency"):
        BatchSimulationRunner(max_total_reflected_efficiency=2.0)  # type: ignore[call-arg]


def test_batch_runner_passes_min_reflected_efficiency_to_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads: list[dict[str, object]] = []

    def fake_run_case_payload(
        payload: dict[str, object],
        *,
        diagnostic_callback: object = None,
    ) -> SingleSimulationResult:
        del diagnostic_callback
        payloads.append(payload)
        return fake_single_result(
            energy_ev=float(payload["energy_ev"]),
            grazing_angle_deg=float(payload["grazing_angle_deg"]),
        )

    monkeypatch.setattr(simulation_batch_module, "_run_case_payload", fake_run_case_payload)
    runner = BatchSimulationRunner(min_reflected_efficiency=-0.125)

    list(
        runner.run_cases(
            [{"case_id": "case-1", "grating": build_test_grating(), "energy_ev": 100.0, "grazing_angle_deg": 4.0}]
        )
    )

    assert len(payloads) == 1
    assert payloads[0]["min_reflected_efficiency"] == pytest.approx(-0.125)
    assert payloads[0]["max_total_reflected_efficiency"] == pytest.approx(1.05)


def test_multilayer_theta_search_auto_tracks_previous_for_dense_steps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    initial_angles: list[float | None] = []

    def fake_run_multilayer_theta_search(**kwargs: object) -> SingleSimulationResult:
        initial_angles.append(
            None if kwargs.get("initial_grazing_angle_deg") is None else float(kwargs["initial_grazing_angle_deg"])
        )
        energy_ev = float(kwargs["energy_ev"])
        theta = 0.50 if energy_ev == 1800.0 else 0.49
        return fake_single_result(
            energy_ev=energy_ev,
            grazing_angle_deg=theta,
            selected_efficiency=0.4,
        )

    monkeypatch.setattr(simulation_module, "run_multilayer_theta_search", fake_run_multilayer_theta_search)
    sweep = run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0, 1802.0],
        output_dir=tmp_path,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
        theta_tracking_mode="auto",
    )

    assert initial_angles == [None, pytest.approx(0.5)]
    second = sweep.batch_result.cases[1]
    assert second.theta_tracking_center_mode == "tracked_previous"
    assert second.theta_tracking_auto_classification == "auto_dense"
    assert second.theta_tracking_used_previous_theta is True
    assert second.theta_tracking_previous_energy_ev == pytest.approx(1800.0)
    assert second.theta_tracking_previous_grazing_angle_deg == pytest.approx(0.5)


def test_multilayer_theta_search_auto_uses_bragg_for_sparse_steps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    initial_angles: list[float | None] = []

    def fake_run_multilayer_theta_search(**kwargs: object) -> SingleSimulationResult:
        initial_angles.append(
            None if kwargs.get("initial_grazing_angle_deg") is None else float(kwargs["initial_grazing_angle_deg"])
        )
        return fake_single_result(
            energy_ev=float(kwargs["energy_ev"]),
            grazing_angle_deg=0.5,
            selected_efficiency=0.4,
        )

    monkeypatch.setattr(simulation_module, "run_multilayer_theta_search", fake_run_multilayer_theta_search)
    sweep = run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0, 1802.0, 1804.0, 2300.0],
        output_dir=tmp_path,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
        theta_tracking_mode="auto",
    )

    assert initial_angles[-1] is None
    last = sweep.batch_result.cases[-1]
    assert last.theta_tracking_center_mode == "bragg"
    assert last.theta_tracking_auto_classification == "auto_sparse"
    assert last.theta_tracking_used_previous_theta is False


def test_multilayer_theta_search_tracking_override_uses_previous(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    initial_angles: list[float | None] = []

    def fake_run_multilayer_theta_search(**kwargs: object) -> SingleSimulationResult:
        initial_angles.append(
            None if kwargs.get("initial_grazing_angle_deg") is None else float(kwargs["initial_grazing_angle_deg"])
        )
        energy_ev = float(kwargs["energy_ev"])
        theta = 0.50 if energy_ev == 1800.0 else 0.49
        return fake_single_result(
            energy_ev=energy_ev,
            grazing_angle_deg=theta,
            selected_efficiency=0.4,
        )

    monkeypatch.setattr(simulation_module, "run_multilayer_theta_search", fake_run_multilayer_theta_search)
    sweep = run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0, 2300.0],
        output_dir=tmp_path,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
        theta_tracking_mode="auto",
        max_tracking_energy_step_ev=1000.0,
    )

    assert initial_angles == [None, pytest.approx(0.5)]
    second = sweep.batch_result.cases[1]
    assert second.theta_tracking_center_mode == "tracked_previous"
    assert second.theta_tracking_auto_classification == "auto_dense"


def test_multilayer_theta_search_tracked_branch_falls_back_to_bragg(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[float, float | None]] = []

    def fake_run_multilayer_theta_search(**kwargs: object) -> SingleSimulationResult:
        energy_ev = float(kwargs["energy_ev"])
        initial = None if kwargs.get("initial_grazing_angle_deg") is None else float(kwargs["initial_grazing_angle_deg"])
        calls.append((energy_ev, initial))
        if energy_ev == 1800.0:
            return fake_single_result(energy_ev=energy_ev, grazing_angle_deg=0.50, selected_efficiency=0.4)
        if initial is not None:
            return fake_single_result(energy_ev=energy_ev, grazing_angle_deg=0.70, selected_efficiency=5e-5)
        return fake_single_result(energy_ev=energy_ev, grazing_angle_deg=0.49, selected_efficiency=0.3)

    monkeypatch.setattr(simulation_module, "run_multilayer_theta_search", fake_run_multilayer_theta_search)
    sweep = run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0, 1802.0],
        output_dir=tmp_path,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
        theta_tracking_mode="auto",
        retry_on_selected_efficiency_zero=False,
        retry_selected_efficiency_threshold=1e-4,
    )

    assert calls == [(1800.0, None), (1802.0, pytest.approx(0.5)), (1802.0, None)]
    second = sweep.batch_result.cases[1]
    assert second.grazing_angle_deg == pytest.approx(0.49)
    assert second.selected_efficiency == pytest.approx(0.3)
    assert second.theta_tracking_center_mode == "bragg"
    assert second.theta_tracking_bragg_fallback_triggered is True


def test_multilayer_theta_search_continuity_guard_rejects_upward_jump(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run_multilayer_theta_search(**kwargs: object) -> SingleSimulationResult:
        energy_ev = float(kwargs["energy_ev"])
        initial = None if kwargs.get("initial_grazing_angle_deg") is None else float(kwargs["initial_grazing_angle_deg"])
        if energy_ev == 1800.0:
            return fake_single_result(energy_ev=energy_ev, grazing_angle_deg=0.50, selected_efficiency=0.4)
        if initial is not None:
            return fake_single_result(energy_ev=energy_ev, grazing_angle_deg=0.60, selected_efficiency=0.39)
        return fake_single_result(energy_ev=energy_ev, grazing_angle_deg=0.49, selected_efficiency=0.38)

    monkeypatch.setattr(simulation_module, "run_multilayer_theta_search", fake_run_multilayer_theta_search)
    sweep = run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0, 1802.0],
        output_dir=tmp_path,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
        theta_tracking_mode="auto",
        retry_on_selected_efficiency_zero=False,
    )

    second = sweep.batch_result.cases[1]
    assert second.grazing_angle_deg == pytest.approx(0.49)
    assert second.theta_tracking_bragg_fallback_triggered is True
    assert second.theta_tracking_continuity_rejected is True


def test_multilayer_theta_search_bragg_mode_matches_legacy_centering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    initial_angles: list[float | None] = []

    def fake_run_multilayer_theta_search(**kwargs: object) -> SingleSimulationResult:
        initial_angles.append(
            None if kwargs.get("initial_grazing_angle_deg") is None else float(kwargs["initial_grazing_angle_deg"])
        )
        return fake_single_result(
            energy_ev=float(kwargs["energy_ev"]),
            grazing_angle_deg=0.5,
            selected_efficiency=0.4,
        )

    monkeypatch.setattr(simulation_module, "run_multilayer_theta_search", fake_run_multilayer_theta_search)
    sweep = run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0, 1802.0],
        output_dir=tmp_path,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
        theta_tracking_mode="bragg",
    )

    assert initial_angles == [None, None]
    assert sweep.batch_result.cases[1].theta_tracking_center_mode == "bragg"


def test_multilayer_theta_search_parallel_auto_uses_multiple_workers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    submitted_payloads: list[dict[str, object]] = []

    class FakeFuture:
        def __init__(self, result: dict[str, object]) -> None:
            self._result = result

        def result(self) -> dict[str, object]:
            return self._result

        def cancel(self) -> None:
            return None

    class FakeExecutor:
        def __init__(self, *, max_workers: int, mp_context: object, initializer: object) -> None:
            self.max_workers = max_workers

        def __enter__(self) -> FakeExecutor:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def submit(self, fn: object, payload: dict[str, object]) -> FakeFuture:
            del fn
            submitted_payloads.append(dict(payload))
            result = simulation_module._single_result_to_record(
                fake_single_result(
                    energy_ev=float(payload["energy_ev"]),
                    grazing_angle_deg=0.5,
                    selected_efficiency=0.4,
                )
            )
            return FakeFuture({"success": True, "result": result})

    monkeypatch.setattr(simulation_module.concurrent.futures, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(simulation_module.concurrent.futures, "as_completed", lambda futures: iter(list(futures)))

    run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0, 1802.0, 1804.0],
        output_dir=tmp_path,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
        theta_tracking_mode="auto",
        max_workers=3,
    )

    assert len(submitted_payloads) == 3
    assert submitted_payloads[0].get("initial_grazing_angle_deg") is None


def test_multilayer_theta_search_parallel_auto_calibrates_first_case_before_pool_submit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[float] = []
    submitted_payloads: list[dict[str, object]] = []
    calibrated_inputs: list[tuple[int, int | None]] = []
    executor_max_workers: list[int] = []

    def fake_run_multilayer_theta_search(**kwargs: object) -> SingleSimulationResult:
        calls.append(float(kwargs["energy_ev"]))
        return fake_single_result(
            energy_ev=float(kwargs["energy_ev"]),
            grazing_angle_deg=0.5,
            selected_efficiency=0.4,
        )

    class FakeFuture:
        def __init__(self, result: dict[str, object]) -> None:
            self._result = result

        def result(self) -> dict[str, object]:
            return self._result

        def cancel(self) -> None:
            return None

    class FakeExecutor:
        def __init__(self, *, max_workers: int, mp_context: object, initializer: object) -> None:
            del mp_context, initializer
            executor_max_workers.append(max_workers)

        def __enter__(self) -> FakeExecutor:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def submit(self, fn: object, payload: dict[str, object]) -> FakeFuture:
            del fn
            submitted_payloads.append(dict(payload))
            result = simulation_module._single_result_to_record(
                fake_single_result(
                    energy_ev=float(payload["energy_ev"]),
                    grazing_angle_deg=0.5,
                    selected_efficiency=0.4,
                )
            )
            return FakeFuture({"success": True, "result": result})

    def fake_calibrate_auto_max_workers_from_result(
        *,
        pending_case_count: int,
        available_memory_bytes: int | None,
    ) -> int:
        calibrated_inputs.append((pending_case_count, available_memory_bytes))
        return 2

    monkeypatch.setattr(simulation_module, "run_multilayer_theta_search", fake_run_multilayer_theta_search)
    monkeypatch.setattr(simulation_module, "_available_memory_bytes", lambda: 123456789)
    monkeypatch.setattr(
        simulation_module,
        "_calibrate_auto_max_workers_from_result",
        fake_calibrate_auto_max_workers_from_result,
    )
    monkeypatch.setattr(simulation_module.concurrent.futures, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(simulation_module.concurrent.futures, "as_completed", lambda futures: iter(list(futures)))

    run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0, 1802.0, 1804.0],
        output_dir=tmp_path,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
        theta_tracking_mode="auto",
        max_workers="auto",
    )

    assert calls == [1800.0]
    assert [float(payload["energy_ev"]) for payload in submitted_payloads] == [1802.0, 1804.0]
    assert calibrated_inputs == [(3, 123456789)]
    assert executor_max_workers == [2]


def test_multilayer_theta_search_parallel_auto_falls_back_when_no_lower_result_available(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    submitted_payloads: list[dict[str, object]] = []

    class FakeFuture:
        def __init__(self, result: dict[str, object]) -> None:
            self._result = result

        def result(self) -> dict[str, object]:
            return self._result

        def cancel(self) -> None:
            return None

    class FakeExecutor:
        def __init__(self, *, max_workers: int, mp_context: object, initializer: object) -> None:
            self.max_workers = max_workers

        def __enter__(self) -> FakeExecutor:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def submit(self, fn: object, payload: dict[str, object]) -> FakeFuture:
            del fn
            submitted_payloads.append(dict(payload))
            result = simulation_module._single_result_to_record(
                fake_single_result(
                    energy_ev=float(payload["energy_ev"]),
                    grazing_angle_deg=0.5,
                    selected_efficiency=0.4,
                )
            )
            return FakeFuture({"success": True, "result": result})

    monkeypatch.setattr(simulation_module.concurrent.futures, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(simulation_module.concurrent.futures, "as_completed", lambda futures: iter(list(futures)))

    run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0, 1802.0],
        output_dir=tmp_path,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
        theta_tracking_mode="auto",
        max_workers=2,
    )

    assert submitted_payloads[1].get("initial_grazing_angle_deg") is None


def test_multilayer_theta_search_parallel_auto_uses_available_lower_result_for_later_dense_point(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    submitted_payloads: list[dict[str, object]] = []

    class FakeFuture:
        def __init__(self, result: dict[str, object], order: int) -> None:
            self._result = result
            self.order = order

        def result(self) -> dict[str, object]:
            return self._result

        def cancel(self) -> None:
            return None

    class FakeExecutor:
        def __init__(self, *, max_workers: int, mp_context: object, initializer: object) -> None:
            self.max_workers = max_workers
            self.counter = 0

        def __enter__(self) -> FakeExecutor:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def submit(self, fn: object, payload: dict[str, object]) -> FakeFuture:
            del fn
            submitted_payloads.append(dict(payload))
            self.counter += 1
            result = simulation_module._single_result_to_record(
                fake_single_result(
                    energy_ev=float(payload["energy_ev"]),
                    grazing_angle_deg=0.5 if float(payload["energy_ev"]) == 1800.0 else 0.49,
                    selected_efficiency=0.4,
                )
            )
            return FakeFuture({"success": True, "result": result}, self.counter)

    def fake_as_completed(futures: object) -> Iterator[FakeFuture]:
        future_list = list(futures)
        future_list.sort(key=lambda future: future.order)
        return iter(future_list)

    monkeypatch.setattr(simulation_module.concurrent.futures, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(simulation_module.concurrent.futures, "as_completed", fake_as_completed)

    run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0, 1802.0, 1804.0],
        output_dir=tmp_path,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
        theta_tracking_mode="auto",
        max_workers=2,
    )

    assert submitted_payloads[2].get("initial_grazing_angle_deg") == pytest.approx(0.5)


def test_multilayer_theta_search_parallel_auto_voigt_calibration_runs_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[float] = []
    submitted_payloads: list[dict[str, object]] = []

    def fake_run_multilayer_theta_search(**kwargs: object) -> SingleSimulationResult:
        calls.append(float(kwargs["energy_ev"]))
        diagnostics = simulation_module.ThetaSearchDiagnostics(
            estimated_grazing_angle_deg=1.2,
            rough_grazing_angles_deg=np.asarray([1.1, 1.2, 1.3], dtype=float),
            rough_efficiencies=np.asarray([0.2, 0.3, 0.25], dtype=float),
            precise_grazing_angles_deg=np.asarray([1.18, 1.2, 1.22], dtype=float),
            precise_efficiencies=np.asarray([0.31, 0.35, 0.33], dtype=float),
            selected_grazing_angle_deg=1.2,
            selected_efficiency=0.35,
            precise_peak_selection_mode_used="voigt",
            precise_peak_fitted_theta_deg=np.asarray([1.18, 1.2, 1.22], dtype=float),
            precise_peak_fitted_efficiencies=np.asarray([0.30, 0.36, 0.32], dtype=float),
        )
        result = fake_single_result(
            energy_ev=float(kwargs["energy_ev"]),
            grazing_angle_deg=1.2,
            selected_efficiency=0.35,
        )
        result.theta_search_diagnostics = diagnostics
        return result

    class FakeFuture:
        def __init__(self, result: dict[str, object]) -> None:
            self._result = result

        def result(self) -> dict[str, object]:
            return self._result

        def cancel(self) -> None:
            return None

    class FakeExecutor:
        def __init__(self, *, max_workers: int, mp_context: object, initializer: object) -> None:
            del max_workers, mp_context, initializer

        def __enter__(self) -> FakeExecutor:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def submit(self, fn: object, payload: dict[str, object]) -> FakeFuture:
            del fn
            submitted_payloads.append(dict(payload))
            result = simulation_module._single_result_to_record(
                fake_single_result(
                    energy_ev=float(payload["energy_ev"]),
                    grazing_angle_deg=1.2,
                    selected_efficiency=0.35,
                )
            )
            return FakeFuture({"success": True, "result": result})

    monkeypatch.setattr(simulation_module, "run_multilayer_theta_search", fake_run_multilayer_theta_search)
    monkeypatch.setattr(simulation_module, "_available_memory_bytes", lambda: 123456789)
    monkeypatch.setattr(simulation_module, "_calibrate_auto_max_workers_from_result", lambda **kwargs: 2)
    monkeypatch.setattr(simulation_module.concurrent.futures, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(simulation_module.concurrent.futures, "as_completed", lambda futures: iter(list(futures)))

    sweep = run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0, 1802.0, 1804.0],
        output_dir=tmp_path,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
        theta_tracking_mode="auto",
        precise_peak_selection_mode="voigt",
        max_workers="auto",
    )

    assert calls == [1800.0]
    assert [float(payload["energy_ev"]) for payload in submitted_payloads] == [1802.0, 1804.0]
    assert [case.energy_ev for case in sweep.batch_result.cases if case.status == "ok"] == [1800.0, 1802.0, 1804.0]


def test_multilayer_theta_search_progress_updates_only_on_completed_points(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    updates: list[int] = []
    postfixes: list[str] = []

    class DummyProgress:
        def __init__(self, total: int | None, desc: str, unit: str) -> None:
            self.total = total
            self.desc = desc
            self.unit = unit

        def update(self, value: int = 1) -> None:
            updates.append(value)

        def set_postfix_str(self, value: str) -> None:
            postfixes.append(value)

        def close(self) -> None:
            return None

    class FakeFuture:
        def __init__(self, result: dict[str, object]) -> None:
            self._result = result

        def result(self) -> dict[str, object]:
            return self._result

        def cancel(self) -> None:
            return None

    class FakeExecutor:
        def __init__(self, *, max_workers: int, mp_context: object, initializer: object) -> None:
            self.max_workers = max_workers

        def __enter__(self) -> FakeExecutor:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def submit(self, fn: object, payload: dict[str, object]) -> FakeFuture:
            del fn
            result = simulation_module._single_result_to_record(
                fake_single_result(
                    energy_ev=float(payload["energy_ev"]),
                    grazing_angle_deg=0.5,
                    selected_efficiency=0.4,
                )
            )
            return FakeFuture({"success": True, "result": result})

    monkeypatch.setattr(simulation_module, "tqdm", DummyProgress)
    monkeypatch.setattr(simulation_module.concurrent.futures, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(simulation_module.concurrent.futures, "as_completed", lambda futures: iter(list(futures)))

    run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0, 1802.0, 1804.0],
        output_dir=tmp_path,
        show_progress=True,
        save_profile_plot=False,
        save_stack_plot=False,
        theta_tracking_mode="auto",
        max_workers=2,
    )

    assert updates == [1, 1, 1]
    assert postfixes
    assert any("active=" in value and "queued=" in value and "done=" in value for value in postfixes)


def test_multilayer_theta_search_sweep_resume_skips_completed_points(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[float] = []

    def fake_run_multilayer_theta_search(**kwargs: object) -> SingleSimulationResult:
        calls.append(float(kwargs["energy_ev"]))
        diagnostics = simulation_module.ThetaSearchDiagnostics(
            estimated_grazing_angle_deg=1.2,
            rough_grazing_angles_deg=np.asarray([1.1, 1.2, 1.3], dtype=float),
            rough_efficiencies=np.asarray([0.2, 0.3, 0.25], dtype=float),
            precise_grazing_angles_deg=np.asarray([1.18, 1.2, 1.22], dtype=float),
            precise_efficiencies=np.asarray([0.31, 0.35, 0.33], dtype=float),
            selected_grazing_angle_deg=1.2,
            selected_efficiency=0.35,
            precise_fwhm_deg=0.04,
            precise_peak_selection_mode_used="voigt",
            precise_peak_fitted_theta_deg=np.asarray([1.18, 1.2, 1.22], dtype=float),
            precise_peak_fitted_efficiencies=np.asarray([0.30, 0.36, 0.32], dtype=float),
        )
        result = fake_single_result(
            energy_ev=float(kwargs["energy_ev"]),
            grazing_angle_deg=1.2,
            selected_efficiency=0.35,
        )
        result.theta_search_diagnostics = diagnostics
        return result

    monkeypatch.setattr(simulation_module, "run_multilayer_theta_search", fake_run_multilayer_theta_search)
    checkpoint_dir = tmp_path / "checkpoints"
    first = run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0, 1802.0],
        output_dir=tmp_path / "first",
        checkpoint_dir=checkpoint_dir,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
    )
    resumed = run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0, 1802.0, 1804.0],
        output_dir=tmp_path / "second",
        checkpoint_dir=checkpoint_dir,
        resume=True,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
    )

    assert len(first.batch_result.cases) == 2
    assert calls == [1800.0, 1802.0, 1804.0]
    assert [case.energy_ev for case in resumed.batch_result.cases if case.status == "ok"] == [1800.0, 1802.0, 1804.0]
    assert (resumed.theta_scan_directory / "theta_scan_1800eV.png").exists()
    assert (resumed.theta_scan_directory / "theta_scan_1802eV.png").exists()
    assert (resumed.theta_scan_directory / "theta_scan_1804eV.png").exists()


def test_multilayer_theta_search_sweep_resume_rebuilds_previous_theta_tracking(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    initial_angles: list[float | None] = []

    def fake_run_multilayer_theta_search(**kwargs: object) -> SingleSimulationResult:
        initial_angles.append(
            None if kwargs.get("initial_grazing_angle_deg") is None else float(kwargs["initial_grazing_angle_deg"])
        )
        energy_ev = float(kwargs["energy_ev"])
        theta = 0.50 if energy_ev == 1800.0 else 0.49
        diagnostics = simulation_module.ThetaSearchDiagnostics(
            estimated_grazing_angle_deg=1.2,
            rough_grazing_angles_deg=np.asarray([1.1, 1.2, 1.3], dtype=float),
            rough_efficiencies=np.asarray([0.2, 0.3, 0.25], dtype=float),
            precise_grazing_angles_deg=np.asarray([1.18, 1.2, 1.22], dtype=float),
            precise_efficiencies=np.asarray([0.31, 0.35, 0.33], dtype=float),
            selected_grazing_angle_deg=theta,
            selected_efficiency=0.35,
            precise_fwhm_deg=0.04,
        )
        result = fake_single_result(
            energy_ev=energy_ev,
            grazing_angle_deg=theta,
            selected_efficiency=0.35,
        )
        result.theta_search_diagnostics = diagnostics
        return result

    monkeypatch.setattr(simulation_module, "run_multilayer_theta_search", fake_run_multilayer_theta_search)
    checkpoint_dir = tmp_path / "checkpoints"
    run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0],
        output_dir=tmp_path / "first",
        checkpoint_dir=checkpoint_dir,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
        theta_tracking_mode="auto",
    )
    initial_angles.clear()
    resumed = run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0, 1802.0],
        output_dir=tmp_path / "second",
        checkpoint_dir=checkpoint_dir,
        resume=True,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
        theta_tracking_mode="auto",
    )

    assert initial_angles == [pytest.approx(0.5)]
    second = [case for case in resumed.batch_result.cases if case.energy_ev == 1802.0][0]
    assert second.theta_tracking_previous_energy_ev == pytest.approx(1800.0)
    assert second.theta_tracking_previous_grazing_angle_deg == pytest.approx(0.5)


def test_multilayer_theta_search_sweep_auto_resume_does_not_rerun_completed_calibration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[float] = []

    def fake_run_multilayer_theta_search(**kwargs: object) -> SingleSimulationResult:
        calls.append(float(kwargs["energy_ev"]))
        return fake_single_result(
            energy_ev=float(kwargs["energy_ev"]),
            grazing_angle_deg=0.5,
            selected_efficiency=0.4,
        )

    class FakeFuture:
        def __init__(self, result: dict[str, object]) -> None:
            self._result = result

        def result(self) -> dict[str, object]:
            return self._result

        def cancel(self) -> None:
            return None

    class FakeExecutor:
        def __init__(self, *, max_workers: int, mp_context: object, initializer: object) -> None:
            del max_workers, mp_context, initializer

        def __enter__(self) -> FakeExecutor:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def submit(self, fn: object, payload: dict[str, object]) -> FakeFuture:
            del fn
            result = simulation_module._single_result_to_record(
                fake_single_result(
                    energy_ev=float(payload["energy_ev"]),
                    grazing_angle_deg=0.5,
                    selected_efficiency=0.4,
                )
            )
            return FakeFuture({"success": True, "result": result})

    monkeypatch.setattr(simulation_module, "run_multilayer_theta_search", fake_run_multilayer_theta_search)
    monkeypatch.setattr(simulation_module, "_available_memory_bytes", lambda: 123456789)
    monkeypatch.setattr(simulation_module, "_calibrate_auto_max_workers_from_result", lambda **kwargs: 2)
    monkeypatch.setattr(simulation_module.concurrent.futures, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(simulation_module.concurrent.futures, "as_completed", lambda futures: iter(list(futures)))

    checkpoint_dir = tmp_path / "checkpoints"
    first = run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0, 1802.0],
        output_dir=tmp_path / "first",
        checkpoint_dir=checkpoint_dir,
        resume=False,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
        theta_tracking_mode="auto",
        max_workers="auto",
    )
    resumed = run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0, 1802.0, 1804.0],
        output_dir=tmp_path / "second",
        checkpoint_dir=checkpoint_dir,
        resume=True,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
        theta_tracking_mode="auto",
        max_workers="auto",
    )

    assert len(first.batch_result.cases) == 2
    assert calls == [1800.0, 1804.0]
    assert [case.energy_ev for case in resumed.batch_result.cases if case.status == "ok"] == [1800.0, 1802.0, 1804.0]


def test_multilayer_theta_search_sweep_resume_progress_preloads_completed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    updates: list[int] = []

    class DummyProgress:
        def __init__(self, total: int | None, desc: str, unit: str) -> None:
            self.total = total
            self.desc = desc
            self.unit = unit

        def update(self, value: int = 1) -> None:
            updates.append(value)

        def set_postfix_str(self, value: str) -> None:
            return None

        def close(self) -> None:
            return None

    def fake_run_multilayer_theta_search(**kwargs: object) -> SingleSimulationResult:
        diagnostics = simulation_module.ThetaSearchDiagnostics(
            estimated_grazing_angle_deg=1.2,
            rough_grazing_angles_deg=np.asarray([1.1, 1.2, 1.3], dtype=float),
            rough_efficiencies=np.asarray([0.2, 0.3, 0.25], dtype=float),
            precise_grazing_angles_deg=np.asarray([1.18, 1.2, 1.22], dtype=float),
            precise_efficiencies=np.asarray([0.31, 0.35, 0.33], dtype=float),
            selected_grazing_angle_deg=1.2,
            selected_efficiency=0.35,
            precise_fwhm_deg=0.04,
        )
        result = fake_single_result(
            energy_ev=float(kwargs["energy_ev"]),
            grazing_angle_deg=1.2,
            selected_efficiency=0.35,
        )
        result.theta_search_diagnostics = diagnostics
        return result

    monkeypatch.setattr(simulation_module, "tqdm", DummyProgress)
    monkeypatch.setattr(simulation_module, "run_multilayer_theta_search", fake_run_multilayer_theta_search)
    checkpoint_dir = tmp_path / "checkpoints"
    run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0],
        output_dir=tmp_path / "first",
        checkpoint_dir=checkpoint_dir,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
    )
    updates.clear()
    run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0, 1802.0],
        output_dir=tmp_path / "second",
        checkpoint_dir=checkpoint_dir,
        resume=True,
        show_progress=True,
        save_profile_plot=False,
        save_stack_plot=False,
    )

    assert updates == [1, 1]


def test_multilayer_theta_search_sweep_resume_ignores_malformed_checkpoint_row(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[float] = []

    def fake_run_multilayer_theta_search(**kwargs: object) -> SingleSimulationResult:
        calls.append(float(kwargs["energy_ev"]))
        diagnostics = simulation_module.ThetaSearchDiagnostics(
            estimated_grazing_angle_deg=1.2,
            rough_grazing_angles_deg=np.asarray([1.1, 1.2, 1.3], dtype=float),
            rough_efficiencies=np.asarray([0.2, 0.3, 0.25], dtype=float),
            precise_grazing_angles_deg=np.asarray([1.18, 1.2, 1.22], dtype=float),
            precise_efficiencies=np.asarray([0.31, 0.35, 0.33], dtype=float),
            selected_grazing_angle_deg=1.2,
            selected_efficiency=0.35,
            precise_fwhm_deg=0.04,
        )
        result = fake_single_result(
            energy_ev=float(kwargs["energy_ev"]),
            grazing_angle_deg=1.2,
            selected_efficiency=0.35,
        )
        result.theta_search_diagnostics = diagnostics
        return result

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "results.jsonl").write_text('{"broken": \n', encoding="utf-8")
    monkeypatch.setattr(simulation_module, "run_multilayer_theta_search", fake_run_multilayer_theta_search)

    run_multilayer_theta_search_sweep(
        grating=build_blazed_multilayer_angle_parity_grating(),
        energies_ev=[1800.0],
        output_dir=tmp_path / "run",
        checkpoint_dir=checkpoint_dir,
        resume=True,
        show_progress=False,
        save_profile_plot=False,
        save_stack_plot=False,
    )

    assert calls == [1800.0]


