"""SX700 PGM working-range geometry and optical efficiency calculations.

The geometry and adaptive grating sampling originate from the
PGM-Working-Range project's graxpy-backend branch.
"""

from .geometry import calculate_working_range, geometry_for_state, solve_alpha_beta
from .optics import calculate_optical_efficiency
from .parameters import PGMCancelled, SX700Parameters
from .side_view import side_view_geometry
from .summary import dashboard_title, parameter_blocks
from .workflow import WorkingRangeResult, run_working_range

__all__ = [
    "PGMCancelled",
    "SX700Parameters",
    "WorkingRangeResult",
    "calculate_optical_efficiency",
    "calculate_working_range",
    "dashboard_title",
    "geometry_for_state",
    "parameter_blocks",
    "run_working_range",
    "side_view_geometry",
    "solve_alpha_beta",
]
