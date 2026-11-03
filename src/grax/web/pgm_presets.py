"""Ready-made PGM setups, written in the same shape as a saved PGM."""

from __future__ import annotations

from typing import Any

DEFAULT_PGM_PRESET = "sx700-example"

# The SX700 example from the PGM-Working-Range project: a blazed Cr/C multilayer
# grating with a Cr/C multilayer mirror, in the second diffraction order. The Cr
# density is the project's own value, not the catalog's 7.15 g/cm³.
PGM_PRESETS: dict[str, dict[str, Any]] = {
    "sx700-example": {
        "id": "sx700-example",
        "name": "SX700 example (2400 l/mm, Cr/C multilayers)",
        "note": (
            "This example takes about a minute on one worker, and less with more workers."
        ),
        "grating_form": {
            "name": "SX700 example grating",
            "grating_type": "blazed",
            "period_lpermm": "2400",
            "x_resolution_nm": "1.0",
            "z_resolution_nm": "0.1",
            "blaze_angle_deg": "0.9",
            "anti_blaze_angle_deg": "3.25",
            "substrate_material": "Si",
            "stack_type": "multilayer",
            "material_a": "Cr",
            "material_a_density_g_cm3": "7.19",
            "material_b": "C",
            "d_period_nm": "4.8",
            "gamma": str(2.0 / 4.8),
            "n_bilayers": "40",
            "top_material": "C",
        },
        "mirror_form": {
            "mirror_name": "SX700 example mirror",
            "substrate_material": "Si",
            "stack_type": "multilayer",
            "material_a": "Cr",
            "material_a_density_g_cm3": "7.19",
            "material_b": "C",
            "d_period_nm": "4.8",
            "gamma": str(2.0 / 4.8),
            "n_bilayers": "60",
            "top_material": "Cr",
            "top_material_density_g_cm3": "7.19",
        },
        "parameters": {
            "diffraction_order": "2",
            "n_energy": "501",
            "n_c": "450",
            "rcwa_energy_nodes": "20",
            "rcwa_cff_nodes": "10",
            "rcwa_harmonics": "9",
        },
    },
}
