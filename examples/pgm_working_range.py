"""Calculate a small SX700 working-range map with the integrated GraxPy API."""

from __future__ import annotations

import grax
from grax.pgm import SX700Parameters, run_working_range


def main() -> None:
    """Print the strongest total-throughput point in a compact example scan."""
    grating = grax.BlazedGrating(
        period_lpermm=2400,
        blaze_angle_deg=0.9,
        anti_blaze_angle_deg=3.25,
        coating_stack=grax.MultilayerStack(
            substrate_material="Si",
            material_a="Cr",
            material_b="C",
            d_period_nm=4.8,
            gamma=2.0 / 4.8,
            n_bilayers=40,
            top_material="C",
        ),
        x_resolution_nm=1.0,
        z_resolution_nm=0.4,
    )
    mirror = grax.MultilayerStack(
        substrate_material="Si",
        material_a="Cr",
        material_b="C",
        d_period_nm=4.8,
        gamma=2.0 / 4.8,
        n_bilayers=60,
        top_material="Cr",
    )
    settings = SX700Parameters(
        grating=grating,
        mirror_stack=mirror,
        diffraction_order=2,
        energy_min_eV=3500,
        energy_max_eV=4100,
        c_min=3.5,
        c_max=4.3,
        n_energy=41,
        n_c=31,
        rcwa_energy_nodes=12,
        rcwa_cff_nodes=10,
        rcwa_harmonics=5,
        rcwa_workers=1,
    )
    result = run_working_range(settings)
    index = result.optics["total_percent"].argmax()
    c_index, e_index = divmod(int(index), len(result.energies_eV))
    print(
        f"Maximum total throughput: {result.optics['total_percent'][c_index, e_index]:.2f}% "
        f"at {result.energies_eV[e_index]:.1f} eV, CFF {result.cff_values[c_index]:.3f}"
    )


if __name__ == "__main__":
    main()
