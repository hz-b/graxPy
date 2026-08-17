"""Fit one parameter set jointly against four differing measurement conditions.

Runs a short first pass and stops, leaving a checkpoint behind. Step 2 resumes
that checkpoint and extends the run, which is the workflow you want on a real
fit: start it, look at the loss history, then decide how much longer to run.
"""

from __future__ import annotations

from example_config import first_pass_trials
from joint_fit import build_spec, report, solver_argument_parser

from grax_opt import optimize_to_joint_measurements


def main() -> None:
    """Run the first pass of the joint fit."""

    args = solver_argument_parser(__doc__ or "Joint measurement fit").parse_args()

    spec = build_spec(solver=args.solver, total_trials=first_pass_trials, resume=False)
    try:
        result = optimize_to_joint_measurements(spec)
    except ImportError as error:
        print(error)
        print("Install the optional optimizer dependency first: `pip install .[opt]`.")
        raise SystemExit(1) from error

    report(result, solver=args.solver, step="First pass")
    print(
        f"Checkpoint written under {spec['output_dir']}/checkpoint. "
        "Run 2_resume_and_extend.py to continue from it."
    )


if __name__ == "__main__":
    main()
