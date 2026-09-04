"""Stage 1: scan the Ru/B4C bilayer thickness ratio at the selected d-spacing.

Runs :func:`grax.run_gamma_study`. With ``D_SPACING_NM = "auto"`` this reads
``d_suggested_nm`` from ``results/optimization_state.json`` written by stage 0.
The suggested gamma is recorded in the state file for traceability; it is not
applied automatically -- copy it into ``ru_b4c_parameters.CONFIG`` if you want
stage 2 to use it.
"""

from __future__ import annotations

from ru_b4c_parameters import CONFIG

from grax import run_gamma_study

if __name__ == "__main__":
    run_gamma_study(CONFIG)
