"""Shared Ax trial-loop runtime for the measurement-fit optimizers."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .optimize import (
    TrialRecord,
    _complete_ax_trial,
    _import_data_required_exception,
    _import_max_parallelism_exception,
)


@dataclass(frozen=True)
class TrialEvaluation:
    """Outcome of evaluating one optimizer candidate.

    Attributes:
        trial_index: Ax trial index the candidate was generated for.
        parameters: Free parameter values evaluated for the candidate.
        loss: Objective value produced by the evaluation.
        resolved_max_workers: Worker count the evaluation actually used.
        extras: Mode-specific payload such as per-measurement losses and
            cached simulated curves.
    """

    trial_index: int
    parameters: dict[str, float]
    loss: float
    resolved_max_workers: int
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialLoopState:
    """Mutable run state shared across optimizer trial iterations.

    Attributes:
        trial_records: Completed trial records in completion order.
        best_loss: Best objective value observed so far.
        best_parameters: Free parameters for the best trial.
        best_grating_parameters: Resolved grating parameters for the best trial.
        best_solver_parameters: Resolved solver parameters for the best trial.
        best_extras: Mode-specific payload captured from the best trial.
        completed_trials: Number of trials successfully evaluated.
        trial_index_cursor: Number of candidates drawn from Ax so far.
        no_improvement_trials: Consecutive trials without a significant improvement.
        stopped_early: Whether early stopping ended the run.
        early_stop_reason: Human-readable early-stopping reason, or ``None``.
        resolved_max_workers: Worker count reported by the most recent trial.
    """

    trial_records: list[TrialRecord] = field(default_factory=list)
    best_loss: float = float("inf")
    best_parameters: dict[str, float] = field(default_factory=dict)
    best_grating_parameters: dict[str, object] = field(default_factory=dict)
    best_solver_parameters: dict[str, float | None] = field(default_factory=dict)
    best_extras: dict[str, Any] = field(default_factory=dict)
    completed_trials: int = 0
    trial_index_cursor: int = 0
    no_improvement_trials: int = 0
    stopped_early: bool = False
    early_stop_reason: str | None = None
    resolved_max_workers: int = 1


def is_significant_improvement(
    previous_best_loss: float,
    loss: float,
    minimum_relative_improvement: float,
) -> bool:
    """Return whether a loss improves on the previous best by enough to reset patience.

    Args:
        previous_best_loss: Best objective value before this trial.
        loss: Objective value produced by this trial.
        minimum_relative_improvement: Minimum relative gain that counts as progress.

    Returns:
        ``True`` when the improvement should reset the early-stopping counter.
    """

    if not loss < previous_best_loss:
        return False
    if not math.isfinite(previous_best_loss):
        return True
    if minimum_relative_improvement <= 0.0:
        return True
    denominator = abs(previous_best_loss)
    if denominator == 0.0:
        return True
    return ((previous_best_loss - loss) / denominator) >= minimum_relative_improvement


def _collect_candidates(
    *,
    ax_client: Any,
    state: TrialLoopState,
    total_trials: int,
    batch_size: int,
    max_parallelism_exception: type[BaseException] | None,
    data_required_exception: type[BaseException] | None,
) -> list[tuple[int, dict[str, float]]]:
    """Draw up to one batch of candidates from the Ax client.

    Args:
        ax_client: Ax client generating the candidates.
        state: Mutable loop state whose cursor is advanced per draw.
        total_trials: Cumulative trial budget for the run.
        batch_size: Maximum number of candidates to draw at once.
        max_parallelism_exception: Ax max-parallelism exception type, if importable.
        data_required_exception: Ax data-required exception type, if importable.

    Returns:
        The drawn candidates as ``(trial_index, parameters)`` pairs.

    Raises:
        Exception: Any error from Ax that is not a known back-pressure signal.
    """

    candidates: list[tuple[int, dict[str, float]]] = []
    while len(candidates) < batch_size and state.trial_index_cursor < total_trials:
        try:
            raw_parameters, trial_index = ax_client.get_next_trial()
        except Exception as error:
            if max_parallelism_exception is not None and isinstance(
                error, max_parallelism_exception
            ):
                break
            if data_required_exception is not None and isinstance(error, data_required_exception):
                break
            raise
        parameters = {name: float(value) for name, value in raw_parameters.items()}
        candidates.append((int(trial_index), parameters))
        state.trial_index_cursor += 1
    return candidates


def run_ax_trial_loop(
    *,
    ax_client: Any,
    config: Any,
    state: TrialLoopState,
    evaluate_candidates: Callable[
        [Sequence[tuple[int, Mapping[str, float]]]], list[TrialEvaluation]
    ],
    on_trial_completed: Callable[..., None],
) -> None:
    """Run the Ax ask-and-tell loop shared by the measurement-fit optimizers.

    The loop owns candidate generation, Ax completion, best-so-far tracking, and
    early stopping. Mode-specific evaluation and artifact persistence are supplied
    by the ``evaluate_candidates`` and ``on_trial_completed`` callbacks.

    Args:
        ax_client: Ax client used to generate and complete trials.
        config: Configuration supplying ``total_trials``, ``batch_size``,
            ``objective_name``, ``objective_sem``, and the early-stopping settings.
        state: Mutable loop state, pre-populated when resuming a run.
        evaluate_candidates: Callable evaluating one batch of candidates.
        on_trial_completed: Callable invoked after each completed trial with
            ``evaluation``, ``state``, and ``improved`` keyword arguments.
    """

    max_parallelism_exception = _import_max_parallelism_exception()
    data_required_exception = _import_data_required_exception()
    total_trials = int(config.total_trials)
    minimum_relative_improvement = float(
        getattr(config, "early_stopping_min_relative_improvement", 0.0)
    )

    while state.trial_index_cursor < total_trials:
        candidates = _collect_candidates(
            ax_client=ax_client,
            state=state,
            total_trials=total_trials,
            batch_size=int(config.batch_size),
            max_parallelism_exception=max_parallelism_exception,
            data_required_exception=data_required_exception,
        )
        if not candidates:
            break

        for evaluation in evaluate_candidates(candidates):
            state.resolved_max_workers = int(evaluation.resolved_max_workers)
            _complete_ax_trial(
                ax_client=ax_client,
                config=config,
                trial_index=evaluation.trial_index,
                loss=float(evaluation.loss),
            )
            state.trial_records.append(
                TrialRecord(
                    trial_index=int(evaluation.trial_index),
                    loss=float(evaluation.loss),
                    parameters=dict(evaluation.parameters),
                    extras=_numeric_extras(evaluation.extras),
                )
            )
            state.completed_trials += 1

            previous_best_loss = state.best_loss
            improved = bool(evaluation.loss < previous_best_loss)
            if improved:
                state.best_loss = float(evaluation.loss)
                state.best_parameters = dict(evaluation.parameters)
                state.best_extras = dict(evaluation.extras)
            if is_significant_improvement(
                previous_best_loss,
                float(evaluation.loss),
                minimum_relative_improvement,
            ):
                state.no_improvement_trials = 0
            else:
                state.no_improvement_trials += 1

            on_trial_completed(evaluation=evaluation, state=state, improved=improved)

            if (
                config.enable_early_stopping
                and state.completed_trials >= config.early_stopping_warmup_trials
                and state.no_improvement_trials >= config.early_stopping_patience
            ):
                state.stopped_early = True
                state.early_stop_reason = (
                    "Early stopping triggered after "
                    f"{state.no_improvement_trials} non-improving trials."
                )
                break
        if state.stopped_early:
            break


def _numeric_extras(extras: Mapping[str, Any]) -> dict[str, float]:
    """Return only the scalar entries of a trial extras payload.

    Args:
        extras: Mode-specific payload attached to a trial evaluation.

    Returns:
        The subset of entries that are finite-castable scalars, for CSV output.
    """

    numeric: dict[str, float] = {}
    for name, value in extras.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric[str(name)] = float(value)
    return numeric
