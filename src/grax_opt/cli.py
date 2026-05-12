"""Command-line entrypoint for Ax-based blazed-grating fitting."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import BlazedAxConfig, InitialBlazedGrating, ParameterBounds
from .optimize import optimize_blazed


def _parse_bounds(bounds: list[float] | None) -> ParameterBounds | None:
    """Convert a CLI bounds list into a typed bounds object."""

    if bounds is None:
        return None
    return ParameterBounds(lower=float(bounds[0]), upper=float(bounds[1]))


def _load_optical_constants(path: str | None) -> object | None:
    """Load a DataFrame of optical constants from a text file path.

    Args:
        path: Path to a text file with `Energy(eV)`, `Delta`, and `Beta`.

    Returns:
        pandas DataFrame with grax optical constants, or `None`.
    """

    if path is None:
        return None
    file_path = Path(path)
    dataframe = pd.read_csv(
        file_path,
        skiprows=1,
        sep=r"\s*,\s*|\s+",
        engine="python",
    )
    dataframe.attrs["name"] = file_path.stem
    return dataframe


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""

    parser = argparse.ArgumentParser(description="Fit a blazed grating with Ax Bayesian optimization.")
    parser.add_argument("--measurement-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--period-lpermm", type=float, required=True)
    parser.add_argument("--blaze-angle-deg", type=float, required=True)
    parser.add_argument("--anti-blaze-angle-deg", type=float, default=None)
    parser.add_argument("--substrate-optical-constants", required=True)
    parser.add_argument("--layer-optical-constants", required=True)
    parser.add_argument("--layer-thickness-nm", type=float, default=30.0)
    parser.add_argument("--top-cap-optical-constants", default=None)
    parser.add_argument("--top-cap-thickness-nm", type=float, default=0.0)
    parser.add_argument("--x-resolution-nm", type=float, default=0.1)
    parser.add_argument("--z-resolution-nm", type=float, default=0.1)
    parser.add_argument("--cff", type=float, default=2.5)
    parser.add_argument("--diffraction-order", type=int, default=1)
    parser.add_argument("--fourier-orders", type=int, default=20)
    parser.add_argument("--total-trials", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--optimize-blaze-angle-deg", action="store_true")
    parser.add_argument("--optimize-anti-blaze-angle-deg", action="store_true")
    parser.add_argument("--optimize-top-cap-thickness-nm", action="store_true")
    parser.add_argument("--period-bounds", type=float, nargs=2)
    parser.add_argument("--blaze-angle-bounds", type=float, nargs=2)
    parser.add_argument("--anti-blaze-angle-bounds", type=float, nargs=2)
    parser.add_argument("--top-cap-thickness-bounds", type=float, nargs=2)
    parser.add_argument("--disable-best-fit-plot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI entrypoint."""

    parser = build_argument_parser()
    arguments = parser.parse_args(argv)

    initial_grating = InitialBlazedGrating(
        period_lpermm=arguments.period_lpermm,
        blaze_angle_deg=arguments.blaze_angle_deg,
        anti_blaze_angle_deg=arguments.anti_blaze_angle_deg,
        substrate_material=_load_optical_constants(arguments.substrate_optical_constants),
        layer_material=_load_optical_constants(arguments.layer_optical_constants),
        layer_thickness_nm=arguments.layer_thickness_nm,
        top_cap_material=_load_optical_constants(arguments.top_cap_optical_constants),
        top_cap_thickness_nm=arguments.top_cap_thickness_nm,
        x_resolution_nm=arguments.x_resolution_nm,
        z_resolution_nm=arguments.z_resolution_nm,
    )
    config = BlazedAxConfig(
        initial_grating=initial_grating,
        measurement_path=Path(arguments.measurement_path),
        output_dir=Path(arguments.output_dir),
        cff=arguments.cff,
        diffraction_order=arguments.diffraction_order,
        fourier_orders=arguments.fourier_orders,
        total_trials=arguments.total_trials,
        random_seed=arguments.random_seed,
        optimize_blaze_angle_deg=arguments.optimize_blaze_angle_deg,
        optimize_anti_blaze_angle_deg=arguments.optimize_anti_blaze_angle_deg,
        optimize_top_cap_thickness_nm=arguments.optimize_top_cap_thickness_nm,
        period_lpermm_bounds=_parse_bounds(arguments.period_bounds),
        blaze_angle_deg_bounds=_parse_bounds(arguments.blaze_angle_bounds),
        anti_blaze_angle_deg_bounds=_parse_bounds(arguments.anti_blaze_angle_bounds),
        top_cap_thickness_nm_bounds=_parse_bounds(arguments.top_cap_thickness_bounds),
        save_best_fit_plot=not arguments.disable_best_fit_plot,
    )
    result = optimize_blazed(config)
    print(f"Best loss: {result.best_loss:.6g}")
    print(f"Best result JSON: {result.result_json_path}")
    print(f"Trial history CSV: {result.trial_history_csv_path}")
    if result.best_fit_plot_path is not None:
        print(f"Best-fit plot: {result.best_fit_plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
