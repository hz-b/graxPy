"""Stage 2: scan the blaze angle of the multilayer-coated Ru/B4C grating.

Runs :func:`grax.run_blaze_study`, which builds the multilayer-coated blazed
grating and runs graxPy's internal theta search per energy for each blaze angle,
then records ``blaze_suggested_deg`` in ``results/optimization_state.json``.

``--solver`` overrides ``ru_b4c_parameters.CONFIG.solver`` for this run. The
executable body is guarded because the theta-search sweep spawns worker
processes that re-import this file by path.
"""

from __future__ import annotations

import argparse
import dataclasses

from ru_b4c_parameters import CONFIG

from grax import run_blaze_study


def main() -> None:
    """Parse ``--solver`` and run the blaze study."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solver",
        choices=("rcwa", "neviere"),
        default=CONFIG.solver,
        help="Electromagnetic solver to run. Both compute every diffraction order; "
        "they differ only in how each layer is crossed in z.",
    )
    args = parser.parse_args()
    run_blaze_study(dataclasses.replace(CONFIG, solver=args.solver))


if __name__ == "__main__":
    main()
