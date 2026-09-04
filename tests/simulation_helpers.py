from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from grax.gratings import BlazedGrating, LaminarGrating
from grax.simulation import (
    SingleSimulationResult,
)
from grax.stacks import MultilayerStack
from tests.optical_constants import load_optical_constants_table

OPTICAL_CONSTANTS_DIR = Path(__file__).resolve().parents[1] / "validation" / "optical_constants"
SI = load_optical_constants_table(OPTICAL_CONSTANTS_DIR / "n_Si_cxro.txt", "Si")
PT = load_optical_constants_table(OPTICAL_CONSTANTS_DIR / "n_Pt_cxro.txt", "Pt")
C = load_optical_constants_table(OPTICAL_CONSTANTS_DIR / "n_C_cxro.txt", "C")
CR = load_optical_constants_table(OPTICAL_CONSTANTS_DIR / "n_Cr_cxro.txt", "Cr")
AU = load_optical_constants_table(OPTICAL_CONSTANTS_DIR / "n_Au_cxro.txt", "Au")

EXAMPLE_SCRIPT_PATHS = [
    Path(__file__).resolve().parents[1] / "examples" / "simulation" / "single_simulation" / "single_simulation.py",
    Path(__file__).resolve().parents[1] / "examples" / "simulation" / "fixed_angle_sweep" / "fixed_angle_sweep.py",
    Path(__file__).resolve().parents[1] / "examples" / "simulation" / "fixed_angle_roughness" / "roughness_kind_comparison.py",
    Path(__file__).resolve().parents[1] / "examples" / "simulation" / "monochromator_sweep" / "monochromator_sweep.py",
    Path(__file__).resolve().parents[1] / "examples" / "simulation" / "energy_angle_sweep" / "energy_angle_sweep.py",
    Path(__file__).resolve().parents[1] / "examples" / "simulation" / "multilayer_theta_search" / "multilayer_theta_search.py",
    Path(__file__).resolve().parents[1] / "examples" / "simulation" / "batch_user_cases" / "batch_user_cases.py",
    Path(__file__).resolve().parents[1] / "examples" / "simulation" / "blazed_multilayer_sweep" / "blazed_multilayer_sweep.py",
    Path(__file__).resolve().parents[1] / "examples" / "simulation" / "blazed_multilayer_memory_comparison" / "blazed_multilayer_memory_comparison.py",
    Path(__file__).resolve().parents[1] / "examples" / "simulation" / "neviere_solver" / "neviere_solver.py",
    Path(__file__).resolve().parents[1] / "examples" / "simulation" / "deep_grating_limits" / "deep_grating_limits.py",
    Path(__file__).resolve().parents[1] / "examples" / "simulation" / "continuous_vs_staircase" / "continuous_vs_staircase.py",
    Path(__file__).resolve().parents[1] / "examples" / "simulation" / "solver_runtime" / "solver_runtime.py",
    Path(__file__).resolve().parents[1]
    / "examples"
    / "simulation"
    / "neviere_grazing_stability"
    / "neviere_grazing_stability.py",
    Path(__file__).resolve().parents[1]
    / "examples"
    / "simulation"
    / "multilayer_optimization_rub4c"
    / "0_ru_b4c_d_spacing_study.py",
    Path(__file__).resolve().parents[1]
    / "examples"
    / "simulation"
    / "multilayer_optimization_rub4c"
    / "1_ru_b4c_gamma_study.py",
    Path(__file__).resolve().parents[1]
    / "examples"
    / "simulation"
    / "multilayer_optimization_rub4c"
    / "2_ru_b4c_blaze_study.py",
]
MULTILAYER_OPT_EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[1] / "examples" / "simulation" / "multilayer_optimization_rub4c"
)
OPTIMIZER_EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[1] / "examples" / "optimizer" / "optimizer_laminar"
)
JOINT_OPTIMIZER_EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[1] / "examples" / "optimizer" / "optimizer_joint"
)


def build_test_grating() -> LaminarGrating:
    """Return a reusable test grating."""
    return LaminarGrating(
        substrate_material=SI,
        layer_material=PT,
        layer_thickness_nm=28.77,
    )


def build_laminar_example_grating(
    *,
    depth_nm: float = 14.9,
    x_resolution_nm: float = 1.0,
    z_resolution_nm: float = 1.0,
) -> LaminarGrating:
    """Return the laminar grating shape used by the public sweep examples."""
    return LaminarGrating(
        period_lpermm=400,
        width_to_period_ratio=0.67,
        depth_nm=depth_nm,
        left_wall_angle_deg=15.0,
        right_wall_angle_deg=15.0,
        substrate_material=SI,
        layer_material=PT,
        layer_thickness_nm=28.77,
        x_resolution_nm=x_resolution_nm,
        z_resolution_nm=z_resolution_nm,
    )


def build_monochromator_example_grating(
    *,
    x_resolution_nm: float = 1.0,
    z_resolution_nm: float = 1.0,
) -> BlazedGrating:
    """Return the blazed single-layer grating used by the public mono example."""
    return BlazedGrating(
        period_lpermm=600,
        substrate_material=SI,
        layer_material=AU,
        layer_thickness_nm=30.0,
        blaze_angle_deg=0.75,
        anti_blaze_angle_deg=5.597,
        x_resolution_nm=x_resolution_nm,
        z_resolution_nm=z_resolution_nm,
    )




def fake_single_result(
    *,
    energy_ev: float = 100.0,
    grazing_angle_deg: float = 4.0,
    orders: np.ndarray | None = None,
    selected_efficiency: float = 0.1,
) -> SingleSimulationResult:
    """Return a small typed single simulation result for batch tests."""
    order_values = np.asarray([-1, 0, 1], dtype=int) if orders is None else orders
    return SingleSimulationResult(
        energy_ev=energy_ev,
        grazing_angle_deg=grazing_angle_deg,
        orders=order_values,
        selected_efficiency=selected_efficiency,
        selected_diffraction_angle_deg=2.0,
        efficiency_all=np.linspace(0.1, 0.3, order_values.size),
        diffraction_angle_all=np.linspace(1.0, 3.0, order_values.size),
        diffraction_order=1,
        fourier_orders=5,
    )


def build_multilayer_parity_grating() -> LaminarGrating:
    """Return the laminar multilayer grating used for Octave parity tests."""
    return LaminarGrating(
        period_lpermm=400,
        width_to_period_ratio=0.67,
        depth_nm=14.9,
        left_wall_angle_deg=15.0,
        right_wall_angle_deg=15.0,
        coating_stack=MultilayerStack(
            substrate_material=SI,
            material_a=CR,
            material_b=C,
            d_period_nm=6.5,
            gamma=0.45,
            n_bilayers=4,
            top_material=C,
        ),
        x_resolution_nm=20.0,
        z_resolution_nm=1.0,
    )


def build_multilayer_solver_regression_grating() -> LaminarGrating:
    """Return the full laminar multilayer grating that previously diverged."""
    return LaminarGrating(
        period_lpermm=400,
        width_to_period_ratio=0.67,
        depth_nm=14.9,
        left_wall_angle_deg=15.0,
        right_wall_angle_deg=15.0,
        coating_stack=MultilayerStack(
            substrate_material=SI,
            material_a=CR,
            material_b=C,
            d_period_nm=6.5,
            gamma=0.45,
            n_bilayers=40,
            top_material=C,
        ),
        x_resolution_nm=1.0,
        z_resolution_nm=0.1,
    )


def build_blazed_multilayer_angle_parity_grating() -> BlazedGrating:
    """Return the blazed multilayer grating used for angle-sweep parity tests."""
    return BlazedGrating(
        period_lpermm=2400,
        blaze_angle_deg=0.9,
        anti_blaze_angle_deg=3.0,
        coating_stack=MultilayerStack(
            substrate_material=SI,
            material_a=CR,
            material_b=C,
            d_period_nm=6.0,
            gamma=0.4,
            n_bilayers=40,
            top_material=C,
        ),
        x_resolution_nm=1.0,
        z_resolution_nm=1.0,
    )


def run_octave_laminar_multilayer_reference(tmp_path: Path) -> dict[str, np.ndarray]:
    """Run the default Octave laminar-multilayer reference fixture."""
    return run_octave_laminar_multilayer_reference_with_parameters(
        tmp_path,
        n_bilayers=4,
        z_resolution_nm=1.0,
        x_resolution_nm=20.0,
        fourier_orders=5,
    )


def run_octave_laminar_multilayer_reference_with_parameters(
    tmp_path: Path,
    *,
    n_bilayers: int,
    z_resolution_nm: float,
    x_resolution_nm: float,
    fourier_orders: int,
) -> dict[str, np.ndarray]:
    """Run the Octave laminar-multilayer reference fixture and load its outputs."""
    if shutil.which("octave-cli") is None:
        pytest.skip("octave-cli is not available.")

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "reticolo" / "tests" / "octave_laminar_multilayer_reference.m"
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    subprocess.run(
        [
            "octave-cli",
            "--quiet",
            str(script_path),
            str(tmp_path),
            str(repo_root),
            str(n_bilayers),
            str(z_resolution_nm),
            str(x_resolution_nm),
            str(fourier_orders),
        ],
        check=True,
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return {
        "x": np.loadtxt(tmp_path / "x.csv", delimiter=","),
        "z": np.loadtxt(tmp_path / "z.csv", delimiter=","),
        "surface": np.loadtxt(tmp_path / "surface.csv", delimiter=","),
        "material_id": np.loadtxt(tmp_path / "material_id.csv", delimiter=","),
        "solver": np.loadtxt(tmp_path / "solver.csv", delimiter=","),
    }


def run_octave_blazed_multilayer_angle_reference(
    tmp_path: Path,
    *,
    start_angle_index: int = 75,
    end_angle_index: int = 81,
) -> np.ndarray:
    """Run the Octave blazed-multilayer angle-sweep reference fixture."""
    if shutil.which("octave-cli") is None:
        pytest.skip("octave-cli is not available.")

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "reticolo" / "tests" / "octave_blazed_multilayer_angle_reference.m"
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    subprocess.run(
        [
            "octave-cli",
            "--quiet",
            str(script_path),
            str(tmp_path),
            str(repo_root),
            str(start_angle_index),
            str(end_angle_index),
        ],
        check=True,
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return np.loadtxt(tmp_path / "solver.csv", delimiter=",")
