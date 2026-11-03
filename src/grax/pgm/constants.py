"""Physical and module-wide constants."""

from grax.planar_mirror import HC_EV_NM

__all__ = ["FALLBACK_GROOVE_DENSITY_L_MM", "HC_EV_NM"]

# Used for geometry-only runs (no grating given and no groove density set).
FALLBACK_GROOVE_DENSITY_L_MM = 2400.0
