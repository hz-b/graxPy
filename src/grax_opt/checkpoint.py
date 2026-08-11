"""Checkpoint and resume support for the measurement-fit optimizers.

A checkpoint directory holds three files:

``ax_client_snapshot.json``
    The Ax client state, so a resumed run keeps its surrogate model instead of
    restarting the generation strategy from scratch.
``optimizer_state.json``
    Run state Ax does not own: best-so-far results, counters, timing metadata,
    and the problem fingerprint guarding against resuming into a changed search
    space.
``trial_records.jsonl``
    Append-only per-trial history. A crash costs at most the final line.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .optimize import TrialRecord

module_logger = logging.getLogger(__name__)

CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class OptimizerCheckpointPaths:
    """Filesystem layout for one optimizer checkpoint directory.

    Attributes:
        checkpoint_dir: Directory holding the checkpoint files.
        ax_snapshot_path: Path to the serialized Ax client state.
        state_path: Path to the optimizer run-state JSON.
        trial_records_path: Path to the append-only trial-record log.
    """

    checkpoint_dir: Path
    ax_snapshot_path: Path
    state_path: Path
    trial_records_path: Path

    @classmethod
    def for_config(cls, config: Any) -> OptimizerCheckpointPaths:
        """Resolve the checkpoint layout for a configuration.

        Args:
            config: Optimizer configuration with ``output_dir`` and an optional
                ``checkpoint_dir``.

        Returns:
            The resolved checkpoint paths.
        """

        checkpoint_dir = getattr(config, "checkpoint_dir", None)
        if checkpoint_dir is None:
            checkpoint_dir = Path(config.output_dir) / "checkpoint"
        checkpoint_dir = Path(checkpoint_dir)
        return cls(
            checkpoint_dir=checkpoint_dir,
            ax_snapshot_path=checkpoint_dir / "ax_client_snapshot.json",
            state_path=checkpoint_dir / "optimizer_state.json",
            trial_records_path=checkpoint_dir / "trial_records.jsonl",
        )

    def exists(self) -> bool:
        """Return whether a usable checkpoint is present.

        Returns:
            ``True`` when both the Ax snapshot and the run state exist.
        """

        return self.ax_snapshot_path.is_file() and self.state_path.is_file()


def _atomic_write_json(output_path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON atomically so a crash cannot truncate the file.

    Args:
        output_path: Destination path to replace.
        payload: JSON-serializable payload.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(output_path.parent),
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _file_content_hash(path: Path) -> str:
    """Return a stable content hash for a measurement file.

    Args:
        path: File to hash.

    Returns:
        The hex digest, or a sentinel when the file is unreadable.
    """

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return "unreadable"


def build_problem_fingerprint(config: Any) -> dict[str, Any]:
    """Describe the parts of a configuration that a resume may not change.

    Settings that only affect run length or performance -- ``total_trials``,
    ``batch_size``, ``max_workers``, ``backend``, artifact flags, and the
    early-stopping settings -- are deliberately excluded so they can be tuned
    between runs.

    Args:
        config: Optimizer configuration to fingerprint.

    Returns:
        A JSON-serializable description of the optimization problem.
    """

    fingerprint: dict[str, Any] = {
        "parameter_bounds": {
            name: [float(bounds.lower), float(bounds.upper)]
            for name, bounds in sorted(config.parameter_bounds.items())
        },
        "equality_constraints": dict(sorted(config.equality_constraints.items())),
        "objective_name": str(config.objective_name),
        "diffraction_order": int(config.diffraction_order),
        "fourier_orders": int(config.fourier_orders),
        "roughness_sigma_nm": config.roughness_sigma_nm,
        "failure_penalty": float(config.failure_penalty),
    }

    measurements = getattr(config, "measurements", None)
    if measurements is not None:
        fingerprint["joint_loss_reduction"] = str(config.joint_loss_reduction)
        fingerprint["measurements"] = [
            {
                "label": str(spec.label),
                "grazing_angle_deg": float(spec.grazing_angle_deg),
                "measurement_path": str(Path(spec.measurement_path).resolve()),
                "content_hash": _file_content_hash(Path(spec.measurement_path)),
                "evaluation_energies_ev": [
                    float(energy_ev) for energy_ev in spec.evaluation_energies_ev
                ],
                "weight": float(spec.weight),
            }
            for spec in measurements
        ]
        return fingerprint

    measurement_path = Path(config.measurement_path)
    fingerprint["measurement_path"] = str(measurement_path.resolve())
    fingerprint["content_hash"] = _file_content_hash(measurement_path)
    fingerprint["angle_mode"] = str(config.angle_mode)
    fingerprint["grazing_angle_deg"] = float(config.grazing_angle_deg)
    fingerprint["cff"] = float(config.cff)
    fingerprint["evaluation_energies_ev"] = [
        float(energy_ev) for energy_ev in config.evaluation_energies_ev
    ]
    fingerprint["evaluation_grazing_angles_deg"] = [
        float(angle) for angle in config.evaluation_grazing_angles_deg
    ]
    return fingerprint


def fingerprint_hash(fingerprint: Mapping[str, Any]) -> str:
    """Return a stable hash for a problem fingerprint.

    Args:
        fingerprint: Fingerprint produced by :func:`build_problem_fingerprint`.

    Returns:
        The hex digest of the canonical JSON encoding.
    """

    canonical = json.dumps(fingerprint, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _describe_fingerprint_differences(
    stored: Mapping[str, Any],
    current: Mapping[str, Any],
) -> list[str]:
    """List the fingerprint keys that differ between two runs.

    Args:
        stored: Fingerprint recorded in the checkpoint.
        current: Fingerprint of the configuration being run now.

    Returns:
        Sorted dotted key paths that differ.
    """

    differences: list[str] = []
    for key in sorted(set(stored) | set(current)):
        stored_value = stored.get(key)
        current_value = current.get(key)
        if stored_value == current_value:
            continue
        if isinstance(stored_value, dict) and isinstance(current_value, dict):
            differences.extend(
                f"{key}.{nested}"
                for nested in _describe_fingerprint_differences(stored_value, current_value)
            )
            continue
        differences.append(key)
    return differences


def verify_fingerprint(
    *,
    stored_fingerprint: Mapping[str, Any],
    current_fingerprint: Mapping[str, Any],
) -> None:
    """Refuse to resume when the optimization problem itself changed.

    Args:
        stored_fingerprint: Fingerprint recorded in the checkpoint.
        current_fingerprint: Fingerprint of the configuration being run now.

    Raises:
        ValueError: If the fingerprints differ, naming the changed keys.
    """

    if stored_fingerprint == current_fingerprint:
        return
    differences = _describe_fingerprint_differences(stored_fingerprint, current_fingerprint)
    raise ValueError(
        "Cannot resume: the checkpoint was created for a different optimization problem "
        f"(changed: {', '.join(differences) or 'unknown'}). Use a different checkpoint_dir "
        "or set resume=False to start a new run."
    )


def append_trial_record(handle: Any, record: TrialRecord) -> None:
    """Append one completed trial to the trial-record log.

    Args:
        handle: Open append-mode file handle.
        record: Completed trial to serialize.
    """

    payload = {
        "trial_index": int(record.trial_index),
        "loss": float(record.loss),
        "parameters": {name: float(value) for name, value in record.parameters.items()},
        "extras": {name: float(value) for name, value in record.extras.items()},
    }
    handle.write(json.dumps(payload) + "\n")


def load_trial_records(trial_records_path: Path) -> list[TrialRecord]:
    """Load completed trial records, tolerating a torn final line.

    Args:
        trial_records_path: Path to the append-only trial-record log.

    Returns:
        The recoverable trial records in file order.
    """

    if not trial_records_path.is_file():
        return []
    records: list[TrialRecord] = []
    with trial_records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
                records.append(
                    TrialRecord(
                        trial_index=int(payload["trial_index"]),
                        loss=float(payload["loss"]),
                        parameters={
                            str(name): float(value)
                            for name, value in payload.get("parameters", {}).items()
                        },
                        extras={
                            str(name): float(value)
                            for name, value in payload.get("extras", {}).items()
                        },
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                module_logger.warning("Ignoring malformed trial record during resume.")
    return records


def load_checkpoint_state(state_path: Path) -> dict[str, Any]:
    """Read the optimizer run-state JSON.

    Args:
        state_path: Path to the run-state file.

    Returns:
        The decoded run state.

    Raises:
        ValueError: If the file is missing or cannot be decoded.
    """

    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Cannot resume: the optimizer state at {state_path} is missing or unreadable "
            f"({error}). Use a different checkpoint_dir or set resume=False to start a new run."
        ) from error


def write_checkpoint_state(
    *,
    paths: OptimizerCheckpointPaths,
    payload: Mapping[str, Any],
) -> None:
    """Persist the optimizer run state atomically.

    Args:
        paths: Resolved checkpoint paths.
        payload: Run state to persist.
    """

    _atomic_write_json(paths.state_path, payload)


def save_ax_client_snapshot(ax_client: Any, snapshot_path: Path) -> None:
    """Persist the Ax client state atomically.

    Args:
        ax_client: Ax client to serialize.
        snapshot_path: Destination snapshot path.
    """

    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = snapshot_path.with_name(f".{snapshot_path.name}.tmp")
    try:
        ax_client.save_to_json_file(filepath=str(temporary_path))
        os.replace(temporary_path, snapshot_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def load_ax_client_snapshot(snapshot_path: Path, *, recorded_ax_version: str | None = None) -> Any:
    """Restore an Ax client from a snapshot.

    Args:
        snapshot_path: Path to the serialized Ax client state.
        recorded_ax_version: Ax version recorded when the snapshot was written.

    Returns:
        The restored Ax client.

    Raises:
        ValueError: If the snapshot cannot be loaded.
    """

    from .optimize import _import_ax_client

    ax_client_cls = _import_ax_client()
    installed_ax_version = _installed_ax_version()
    if recorded_ax_version is not None and recorded_ax_version != installed_ax_version:
        module_logger.warning(
            "Resuming a checkpoint written with ax %s using ax %s.",
            recorded_ax_version,
            installed_ax_version,
        )
    try:
        return ax_client_cls.load_from_json_file(filepath=str(snapshot_path))
    except Exception as error:
        raise ValueError(
            f"Cannot resume: the Ax snapshot at {snapshot_path} could not be loaded "
            f"({type(error).__name__}: {error}). It was written with ax "
            f"{recorded_ax_version or 'unknown'} and this environment has ax "
            f"{installed_ax_version}. Use a different checkpoint_dir or set resume=False "
            "to start a new run."
        ) from error


def _installed_ax_version() -> str:
    """Return the installed Ax version.

    Returns:
        The version string, or ``"unknown"`` when it cannot be determined.
    """

    try:
        import ax
    except ImportError:
        return "unknown"
    return str(getattr(ax, "__version__", "unknown"))


def ax_trial_count(ax_client: Any) -> int | None:
    """Return how many trials the Ax client has already issued.

    Args:
        ax_client: Ax client to inspect.

    Returns:
        The trial count, or ``None`` when it cannot be determined.
    """

    experiment = getattr(ax_client, "experiment", None)
    if experiment is None:
        return None
    try:
        return int(len(experiment.trials))
    except (AttributeError, TypeError):
        return None


def build_checkpoint_state_payload(
    *,
    config: Any,
    state: Any,
    fingerprint: Mapping[str, Any],
    previous_state: Mapping[str, Any] | None,
    backend_requested: str,
    backend_effective: str,
    best_extras_payload: Mapping[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Assemble the optimizer run-state payload.

    Timing metadata follows the batch-sweep convention: ``created`` is preserved
    across resumes and elapsed time accumulates.

    Args:
        config: Optimizer configuration for the current run.
        state: Trial-loop state to persist.
        fingerprint: Problem fingerprint for the current configuration.
        previous_state: Run state loaded at resume, if any.
        backend_requested: Backend requested by the caller.
        backend_effective: Backend actually used.
        best_extras_payload: JSON-safe extras captured from the best trial.
        elapsed_seconds: Wall time spent in the current run.

    Returns:
        The run state to persist.
    """

    now_iso = datetime.now().isoformat()
    previous = dict(previous_state or {})
    total_trials_history = list(previous.get("total_trials_history", []))
    total_trials_history.append(int(config.total_trials))
    cumulative_elapsed = float(previous.get("cumulative_elapsed_seconds", 0.0)) + float(
        elapsed_seconds
    )

    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "ax_version": _installed_ax_version(),
        "fingerprint": dict(fingerprint),
        "fingerprint_hash": fingerprint_hash(fingerprint),
        "best_loss": float(state.best_loss),
        "best_parameters": dict(state.best_parameters),
        "best_extras": dict(best_extras_payload),
        "completed_trials": int(state.completed_trials),
        "trial_index_cursor": int(state.trial_index_cursor),
        "no_improvement_trials": int(state.no_improvement_trials),
        "stopped_early": bool(state.stopped_early),
        "early_stop_reason": state.early_stop_reason,
        "total_trials_history": total_trials_history,
        "random_seed": config.random_seed,
        "backend_requested": backend_requested,
        "backend_effective": backend_effective,
        "optimizer_resolved_max_workers": int(state.resolved_max_workers),
        "created": previous.get("created", now_iso),
        "current_run_started": previous.get("current_run_started_marker", now_iso),
        "last_updated": now_iso,
        "cumulative_elapsed_seconds": cumulative_elapsed,
        "last_run_elapsed_seconds": float(elapsed_seconds),
        "run_count": int(previous.get("run_count", 0)) + 1,
    }


def json_safe_extras(extras: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a trial extras payload into JSON-serializable values.

    Args:
        extras: Mode-specific payload attached to a trial evaluation.

    Returns:
        The payload with arrays converted to lists.
    """

    safe: dict[str, Any] = {}
    for name, value in extras.items():
        if hasattr(value, "tolist"):
            safe[str(name)] = value.tolist()
        elif isinstance(value, Mapping):
            safe[str(name)] = {
                str(inner_name): (
                    inner_value.tolist() if hasattr(inner_value, "tolist") else inner_value
                )
                for inner_name, inner_value in value.items()
            }
        else:
            safe[str(name)] = value
    return safe


def restore_best_extras(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild a best-trial extras payload loaded from JSON.

    Args:
        payload: Extras payload as stored in the checkpoint.

    Returns:
        The payload with simulated curves restored as arrays.
    """

    import numpy as np

    restored = dict(payload)
    simulated = restored.get("simulated_by_label")
    if isinstance(simulated, Mapping):
        restored["simulated_by_label"] = {
            str(label): np.asarray(values, dtype=float) for label, values in simulated.items()
        }
    return restored


class OptimizerCheckpointSession:
    """Persists and restores optimizer progress for one run.

    Checkpoints are always written so an interrupted run can be continued later,
    but they are only read back when the configuration sets ``resume=True``.

    Attributes:
        paths: Resolved checkpoint file layout.
        enabled: Whether checkpoint writing is active.
        resumed: Whether state was restored from an existing checkpoint.
        previous_state: Run state loaded at resume, if any.
    """

    def __init__(
        self,
        *,
        config: Any,
        backend_requested: str,
        backend_effective: str,
    ) -> None:
        """Initialize a checkpoint session for one optimizer run.

        Args:
            config: Optimizer configuration for the run.
            backend_requested: Backend requested by the caller.
            backend_effective: Backend actually used.
        """

        self._config = config
        self._backend_requested = backend_requested
        self._backend_effective = backend_effective
        self.paths = OptimizerCheckpointPaths.for_config(config)
        self.enabled = True
        self.resumed = False
        self.previous_state: dict[str, Any] | None = None
        self._fingerprint = build_problem_fingerprint(config)
        self._handle: Any = None
        self._since_flush = 0
        self._started_monotonic = 0.0

    def restore_or_create_ax_client(self, create_ax_client: Any, state: Any) -> Any:
        """Restore the Ax client from a checkpoint, or create a fresh one.

        Args:
            create_ax_client: Callable creating a new Ax client for the config.
            state: Trial-loop state to populate when resuming.

        Returns:
            The Ax client to drive the run.

        Raises:
            ValueError: If a partial or mismatched checkpoint is found.
        """

        if not bool(getattr(self._config, "resume", False)):
            return create_ax_client(self._config)

        if not self.paths.exists():
            if self.paths.ax_snapshot_path.is_file() or self.paths.state_path.is_file():
                raise ValueError(
                    f"Cannot resume: the checkpoint at {self.paths.checkpoint_dir} is "
                    "incomplete. Use a different checkpoint_dir or set resume=False to "
                    "start a new run."
                )
            module_logger.info(
                "No checkpoint found at %s; starting a new optimization run.",
                self.paths.checkpoint_dir,
            )
            return create_ax_client(self._config)

        checkpoint_state = load_checkpoint_state(self.paths.state_path)
        verify_fingerprint(
            stored_fingerprint=checkpoint_state.get("fingerprint", {}),
            current_fingerprint=self._fingerprint,
        )
        if checkpoint_state.get("random_seed") != self._config.random_seed:
            module_logger.warning(
                "Resuming with random_seed=%s but the checkpoint recorded %s; "
                "already-generated trials are unaffected.",
                self._config.random_seed,
                checkpoint_state.get("random_seed"),
            )

        ax_client = load_ax_client_snapshot(
            self.paths.ax_snapshot_path,
            recorded_ax_version=checkpoint_state.get("ax_version"),
        )
        restore_trial_loop_state(
            state=state,
            checkpoint_state=checkpoint_state,
            trial_records=load_trial_records(self.paths.trial_records_path),
            ax_client=ax_client,
        )
        state.best_extras = restore_best_extras(checkpoint_state.get("best_extras", {}))
        self.resumed = True
        self.previous_state = checkpoint_state
        module_logger.info(
            "Resumed optimization from %s with %s completed trials.",
            self.paths.checkpoint_dir,
            state.completed_trials,
        )
        return ax_client

    def __enter__(self) -> OptimizerCheckpointSession:
        """Open the trial-record log for appending.

        Returns:
            This session.
        """

        import time

        self._started_monotonic = time.perf_counter()
        if self.enabled:
            self.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self._handle = self.paths.trial_records_path.open("a", encoding="utf-8")
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Flush and close the trial-record log.

        Args:
            exc_type: Exception type, if the block raised.
            exc_value: Exception value, if the block raised.
            traceback: Traceback, if the block raised.
        """

        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None

    def record_trial(self, *, state: Any, ax_client: Any) -> None:
        """Append the newest trial and periodically persist full state.

        Args:
            state: Trial-loop state after the trial completed.
            ax_client: Ax client to snapshot.
        """

        if not self.enabled or self._handle is None or not state.trial_records:
            return
        append_trial_record(self._handle, state.trial_records[-1])
        self._since_flush += 1
        if self._since_flush >= int(self._config.checkpoint_interval):
            self._handle.flush()
            self._since_flush = 0
            self.persist(state=state, ax_client=ax_client)

    def persist(self, *, state: Any, ax_client: Any) -> None:
        """Write the Ax snapshot and optimizer run state atomically.

        Args:
            state: Trial-loop state to persist.
            ax_client: Ax client to snapshot.
        """

        if not self.enabled:
            return
        import time

        if self._handle is not None:
            self._handle.flush()
        try:
            save_ax_client_snapshot(ax_client, self.paths.ax_snapshot_path)
        except Exception as error:
            module_logger.warning(
                "Could not write the Ax checkpoint snapshot: %s: %s.",
                type(error).__name__,
                error,
            )
            return
        write_checkpoint_state(
            paths=self.paths,
            payload=build_checkpoint_state_payload(
                config=self._config,
                state=state,
                fingerprint=self._fingerprint,
                previous_state=self.previous_state,
                backend_requested=self._backend_requested,
                backend_effective=self._backend_effective,
                best_extras_payload=json_safe_extras(state.best_extras),
                elapsed_seconds=time.perf_counter() - self._started_monotonic,
            ),
        )


def restore_trial_loop_state(
    *,
    state: Any,
    checkpoint_state: Mapping[str, Any],
    trial_records: Sequence[TrialRecord],
    ax_client: Any,
) -> None:
    """Restore in-memory loop state from a checkpoint.

    Ax is authoritative for how many candidates have been issued, so the cursor
    is taken from the client and reconciled against the recovered trial records.

    Args:
        state: Trial-loop state to populate in place.
        checkpoint_state: Decoded optimizer run state.
        trial_records: Trial records recovered from the log.
        ax_client: Restored Ax client.
    """

    state.trial_records = list(trial_records)
    state.completed_trials = len(state.trial_records)
    state.best_loss = float(checkpoint_state.get("best_loss", float("inf")))
    state.best_parameters = dict(checkpoint_state.get("best_parameters", {}))
    state.no_improvement_trials = int(checkpoint_state.get("no_improvement_trials", 0))
    state.resolved_max_workers = int(
        checkpoint_state.get("optimizer_resolved_max_workers", state.resolved_max_workers)
    )

    persisted_cursor = int(checkpoint_state.get("trial_index_cursor", state.completed_trials))
    client_cursor = ax_trial_count(ax_client)
    if client_cursor is None:
        state.trial_index_cursor = persisted_cursor
        return
    if client_cursor != state.completed_trials:
        module_logger.warning(
            "Checkpoint drift on resume: Ax reports %s issued trials but %s trial records "
            "were recovered. Continuing from the larger count.",
            client_cursor,
            state.completed_trials,
        )
    state.trial_index_cursor = max(client_cursor, persisted_cursor)
