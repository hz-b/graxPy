"""Resume the checkpointed joint fit and extend it to the full trial budget.

``total_trials`` is cumulative across resumed runs, so raising it from
``first_pass_trials`` to ``total_trials`` asks for the difference, not for a
whole new run. The Ax surrogate model is restored from the checkpoint, so the
extra trials continue the same search instead of restarting it.

The same spec is rebuilt here as in step 1. That is deliberate: the checkpoint
records a fingerprint of the problem -- bounds, constraints, measurement
contents and conditions, solver -- and refuses to resume if any of it changed.
"""

from __future__ import annotations

import json

from example_config import results_dir, total_trials
from joint_fit import build_spec, report, solver_argument_parser

from grax_opt import optimize_to_joint_measurements


def main() -> None:
    """Resume the checkpoint and run the remaining trials."""

    args = solver_argument_parser(__doc__ or "Resume a joint measurement fit").parse_args()

    checkpoint_dir = results_dir(args.solver) / "checkpoint"
    if not (checkpoint_dir / "optimizer_state.json").is_file():
        print(f"No checkpoint at {checkpoint_dir}. Run 1_fit_joint.py first.")
        raise SystemExit(1)

    state_before = json.loads((checkpoint_dir / "optimizer_state.json").read_text(encoding="utf-8"))
    trials_before = int(state_before["completed_trials"])

    spec = build_spec(solver=args.solver, total_trials=total_trials, resume=True)
    try:
        result = optimize_to_joint_measurements(spec)
    except ImportError as error:
        print(error)
        print("Install the optional optimizer dependency first: `pip install .[opt]`.")
        raise SystemExit(1) from error

    # The budget the previous run was given comes from the checkpoint, not from
    # first_pass_trials: after a second resume the earlier budget is whatever
    # total_trials was at the time, which the checkpoint records per run.
    previous_budget = int(state_before["total_trials_history"][-1])

    report(result, solver=args.solver, step="Resumed run")
    print(
        f"Trials before resume: {trials_before} (budget {previous_budget}); "
        f"after: {result.completed_trials} (budget {total_trials}); "
        f"new this run: {result.completed_trials - trials_before}"
    )

    state_after = json.loads((checkpoint_dir / "optimizer_state.json").read_text(encoding="utf-8"))
    print(f"Runs recorded in this checkpoint: {state_after['run_count']}")
    print(f"Trial budgets across runs: {state_after['total_trials_history']}")
    print(f"Cumulative optimizer time: {state_after['cumulative_elapsed_seconds']:.1f} s")


if __name__ == "__main__":
    main()
