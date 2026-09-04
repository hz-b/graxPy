"""Stage 0: derive and scan the Ru/B4C multilayer d-spacing.

Runs :func:`grax.run_d_spacing_study` with the shared ``ru_b4c_parameters.CONFIG``
and writes ``results/optimization_state.json`` with ``d_suggested_nm`` for the
next two stages.
"""

from __future__ import annotations

from ru_b4c_parameters import CONFIG

from grax import run_d_spacing_study

if __name__ == "__main__":
    run_d_spacing_study(CONFIG)
