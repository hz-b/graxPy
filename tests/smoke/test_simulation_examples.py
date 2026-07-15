from __future__ import annotations

import ast
import inspect
import py_compile
from pathlib import Path

from grax.gratings import BlazedGrating
from grax.simulation import (
    BatchSimulationRunner,
    energy_angle_cases,
    fixed_angle_cases,
    monochromator_cases,
    monochromator_grazing_angles_deg,
    multilayer_theta_search_cases,
    plot_order_subset,
    run_multilayer_theta_search_sweep,
    run_simulation,
    write_all_orders_csv,
)
from grax.stacks import MultilayerStack
from tests.simulation_helpers import (
    CR,
    EXAMPLE_SCRIPT_PATHS,
    OPTIMIZER_EXAMPLE_ROOT,
    SI,
    C,
    build_blazed_multilayer_angle_parity_grating,
    build_laminar_example_grating,
    build_monochromator_example_grating,
)


def test_public_examples_do_not_expose_quick_mode_flags() -> None:
    for example_path in EXAMPLE_SCRIPT_PATHS:
        source = example_path.read_text(encoding="utf-8")
        assert "--quick" not in source
        assert "quick_mode" not in source
        assert "Quick mode" not in source


def test_example_and_comparison_scripts_compile_and_use_current_case_helper_kwargs() -> None:
    script_roots = [
        Path(__file__).resolve().parents[2] / "examples",
        Path(__file__).resolve().parents[2] / "validation",
    ]
    script_paths = sorted({path for root in script_roots for path in root.rglob("*.py")})

    for path in script_paths:
        py_compile.compile(str(path), doraise=True)

    helper_functions = {
        "fixed_angle_cases": fixed_angle_cases,
        "monochromator_cases": monochromator_cases,
        "energy_angle_cases": energy_angle_cases,
        "multilayer_theta_search_cases": multilayer_theta_search_cases,
    }
    allowed_kwargs = {
        name: set(inspect.signature(func).parameters)
        for name, func in helper_functions.items()
    }

    unexpected_kwargs: list[str] = []
    for path in script_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            helper_name = None
            if isinstance(node.func, ast.Attribute):
                helper_name = node.func.attr
            elif isinstance(node.func, ast.Name):
                helper_name = node.func.id
            if helper_name not in allowed_kwargs:
                continue
            for keyword in node.keywords:
                if keyword.arg is None:
                    continue
                if keyword.arg not in allowed_kwargs[helper_name]:
                    unexpected_kwargs.append(
                        f"{path}:{node.lineno}:{node.col_offset} -> {helper_name}({keyword.arg})"
                    )

    assert not unexpected_kwargs, "Unexpected helper kwargs found:\n" + "\n".join(unexpected_kwargs)


def test_optimizer_example_assets_exist() -> None:
    expected_paths = [
        OPTIMIZER_EXAMPLE_ROOT / "0_fit_laminar_grating.py",
        OPTIMIZER_EXAMPLE_ROOT / "1_run_simulation_design_parameters.py",
        OPTIMIZER_EXAMPLE_ROOT / "2_run_simulation_fitted_parameters.py",
        OPTIMIZER_EXAMPLE_ROOT / "3_plot_laminar_fit_comparison.py",
        OPTIMIZER_EXAMPLE_ROOT / "measured_alpha4deg_order1.csv",
        OPTIMIZER_EXAMPLE_ROOT / "optical_constants" / "old" / "n_Si_cxro.txt",
        OPTIMIZER_EXAMPLE_ROOT / "optical_constants" / "old" / "n_Pt_cxro.txt",
        OPTIMIZER_EXAMPLE_ROOT / "optical_constants" / "old" / "n_C_cxro.txt",
    ]
    for path in expected_paths:
        assert path.exists(), f"Missing optimizer example asset: {path}"


def test_optimizer_example_scripts_compile() -> None:
    import py_compile

    py_compile.compile(str(OPTIMIZER_EXAMPLE_ROOT / "0_fit_laminar_grating.py"), doraise=True)
    py_compile.compile(str(OPTIMIZER_EXAMPLE_ROOT / "1_run_simulation_design_parameters.py"), doraise=True)
    py_compile.compile(str(OPTIMIZER_EXAMPLE_ROOT / "2_run_simulation_fitted_parameters.py"), doraise=True)
    py_compile.compile(str(OPTIMIZER_EXAMPLE_ROOT / "3_plot_laminar_fit_comparison.py"), doraise=True)


def test_optimizer_example_plot_uses_evaluation_energies() -> None:
    plot_source = (OPTIMIZER_EXAMPLE_ROOT / "3_plot_laminar_fit_comparison.py").read_text(
        encoding="utf-8"
    )
    assert "evaluation_energies_ev" in plot_source
    assert "Optimization energies" in plot_source


def test_multilayer_theta_search_docs_use_grouped_canonical_arguments() -> None:
    example_path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "simulation"
        / "multilayer_theta_search"
        / "multilayer_theta_search.py"
    )
    tutorial_path = Path(__file__).resolve().parents[2] / "docs" / "tutorials" / "multilayer-theta-search.md"

    example_source = example_path.read_text(encoding="utf-8")
    tutorial_source = tutorial_path.read_text(encoding="utf-8")
    example_call = example_source.split("run_multilayer_theta_search_sweep(", maxsplit=1)[1].split(")\n", maxsplit=1)[0]
    tutorial_call = tutorial_source.split("run_multilayer_theta_search_sweep(", maxsplit=1)[1].split(")\n", maxsplit=1)[0]
    assert "run_multilayer_theta_search(" not in tutorial_source

    for call_block in (example_call, tutorial_call):
        assert call_block.index("multilayer_bragg_order") < call_block.index("rough_scan_half_width_deg")
        assert call_block.index("rough_scan_half_width_deg") < call_block.index("rough_fourier_orders")
        assert call_block.index("rough_fourier_orders") < call_block.index("rough_x_resolution_nm")
        assert call_block.index("rough_x_resolution_nm") < call_block.index("fine_scan_half_width_deg")
        assert call_block.index("fine_scan_half_width_deg") < call_block.index("fine_fourier_orders")
        assert call_block.index("fine_fourier_orders") < call_block.index("fine_x_resolution_nm")
        assert call_block.index("fine_x_resolution_nm") < call_block.index("final_fourier_orders")
        assert call_block.index("final_fourier_orders") < call_block.index("final_x_resolution_nm")
        assert call_block.index("final_x_resolution_nm") < call_block.index("precise_peak_selection_mode")
        assert "\n        fourier_orders=" not in call_block
        assert "\n        x_resolution_nm=" not in call_block
        assert "\n        z_resolution_nm=" not in call_block


def test_single_simulation_example_parity_quick_configuration(tmp_path: Path) -> None:
    grating = build_laminar_example_grating(x_resolution_nm=1.0, z_resolution_nm=1.0)
    result = run_simulation(
        grating=grating,
        energy_ev=200.0,
        grazing_angle_deg=4.0,
        diffraction_order=1,
        fourier_orders=3,
    )
    csv_path = tmp_path / "single_simulation.csv"
    profile_path = tmp_path / "single_simulation_profile.png"

    write_all_orders_csv(result, csv_path)
    grating.plot_profile(profile_path)

    assert result.selected_efficiency >= 0.0
    assert csv_path.exists()
    assert profile_path.exists()


def test_fixed_angle_example_parity_quick_configuration(tmp_path: Path) -> None:
    grating = build_laminar_example_grating(x_resolution_nm=1.0, z_resolution_nm=1.0)
    cases = fixed_angle_cases(grating=grating, energies_ev=[200.0], grazing_angle_deg=4.0)
    runner = BatchSimulationRunner(default_diffraction_order=1, default_fourier_orders=3)
    results = list(runner.run_cases(cases))
    csv_path = tmp_path / "fixed_angle_all_orders.csv"
    orders_plot_path = tmp_path / "fixed_angle_orders_1_3.png"

    write_all_orders_csv(results, csv_path)
    plot_order_subset(results, orders_plot_path, diffraction_orders=[1, 2, 3], title="Fixed-angle parity")

    assert len(results) == 1
    assert results[0].status == "ok"
    assert csv_path.exists()
    assert orders_plot_path.exists()


def test_monochromator_example_parity_quick_configuration(tmp_path: Path) -> None:
    grating = build_monochromator_example_grating(x_resolution_nm=1.0, z_resolution_nm=1.0)
    cases = monochromator_cases(grating=grating, energies_ev=[200.0], diffraction_order=1, cff=2.25)
    runner = BatchSimulationRunner(default_fourier_orders=3)
    results = list(runner.run_cases(cases))
    csv_path = tmp_path / "monochromator_all_orders.csv"
    orders_plot_path = tmp_path / "monochromator_orders_1_3.png"

    write_all_orders_csv(results, csv_path)
    plot_order_subset(results, orders_plot_path, diffraction_orders=[1, 2, 3], title="Monochromator parity")

    assert len(results) == 1
    assert results[0].status == "ok"
    assert csv_path.exists()
    assert orders_plot_path.exists()


def test_blazed_multilayer_sweep_example_parity_quick_configuration(tmp_path: Path) -> None:
    grating = BlazedGrating(
        period_lpermm=2400,
        blaze_angle_deg=1.37,
        anti_blaze_angle_deg=3.25,
        coating_stack=MultilayerStack(
            substrate_material=SI,
            material_a=CR,
            material_b=C,
            d_period_nm=6.0,
            gamma=0.4,
            n_bilayers=50,
            top_material=C,
        ),
        x_resolution_nm=1.0,
        z_resolution_nm=1.0,
    )
    cases = monochromator_cases(grating=grating, energies_ev=[500.0], diffraction_order=1, cff=2.25)
    runner = BatchSimulationRunner(default_diffraction_order=1, default_fourier_orders=3)
    results = list(runner.run_cases(cases))
    csv_path = tmp_path / "blazed_multilayer_all_orders.csv"

    write_all_orders_csv(results, csv_path)

    assert len(results) == 1
    assert results[0].status == "ok"
    assert csv_path.exists()


def test_blazed_multilayer_memory_comparison_example_structure() -> None:
    script_path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "simulation"
        / "blazed_multilayer_memory_comparison"
        / "blazed_multilayer_memory_comparison.py"
    )

    source = script_path.read_text(encoding="utf-8")

    assert "grax.MultilayerStack(" in source
    assert "grax.BlazedGrating(" in source
    assert "grax.monochromator_cases(" in source
    assert "grax.BatchSimulationRunner(" in source
    assert "show_progress=True" in source
    assert 'max_workers="auto"' in source
    assert 'sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))' in source
    assert '"memory_mode"' not in source
    assert '"_memory_mode"' not in source
    assert 'profile_memory": True' in source
    assert "blazed_multilayer_memory_comparison.csv" in source
    assert "blazed_multilayer_memory_comparison.png" in source
    assert "blazed_multilayer_profile.png" in source
    assert "multilayer_stack_schematic.png" in source


def test_energy_angle_example_parity_quick_configuration(tmp_path: Path) -> None:
    grating = build_blazed_multilayer_angle_parity_grating()
    cases = energy_angle_cases(grating=grating, energy_angle_pairs=[(1800.0, 8.0)])
    runner = BatchSimulationRunner(default_diffraction_order=2, default_fourier_orders=3)
    results = list(runner.run_cases(cases))
    csv_path = tmp_path / "energy_angle_all_orders.csv"

    write_all_orders_csv(results, csv_path)

    assert len(results) == 1
    assert results[0].status == "ok"
    assert csv_path.exists()


def test_multilayer_theta_search_example_parity_quick_configuration(tmp_path: Path) -> None:
    grating = build_blazed_multilayer_angle_parity_grating()
    sweep = run_multilayer_theta_search_sweep(
        grating=grating,
        energies_ev=[1800.0],
        output_dir=tmp_path,
        diffraction_order=2,
        multilayer_bragg_order=1,
        rough_scan_half_width_deg=0.5,
        rough_scan_points=21,
        fine_scan_half_width_deg=0.1,
        fine_scan_points=21,
        rough_fourier_orders=3,
        fine_fourier_orders=3,
        final_fourier_orders=3,
        rough_x_resolution_nm=1.0,
        rough_z_resolution_nm=1.0,
        fine_x_resolution_nm=1.0,
        fine_z_resolution_nm=1.0,
        final_x_resolution_nm=1.0,
        final_z_resolution_nm=1.0,
        precise_peak_selection_mode="voigt",
        retry_on_selected_efficiency_zero=True,
        retry_selected_efficiency_threshold=1e-3,
        max_zero_efficiency_retries=1,
        show_progress=False,
        live_plot=False,
        on_error="fail_fast",
        save_profile_plot=False,
        save_stack_plot=False,
        backend="numba",
    )

    assert len(sweep.batch_result.cases) == 1
    assert sweep.batch_result.cases[0].status == "ok"
    assert sweep.summary_csv_path.exists()
    assert sweep.theta_scan_directory.exists()


def test_batch_user_cases_example_parity_quick_configuration(tmp_path: Path) -> None:
    grating = build_laminar_example_grating(depth_nm=14.9, x_resolution_nm=1.0, z_resolution_nm=1.0)
    grazing_angle_deg = float(
        monochromator_grazing_angles_deg(
            [1000.0],
            period_lpermm=grating.period_lpermm,
            diffraction_order=1,
            cff=2.25,
        )[0]
    )
    user_cases = [
        {
            "case_id": "user-laminar-depth-015",
            "label": "Laminar grating at depth 14.9 nm",
            "grating": grating,
            "energy_ev": 1000.0,
            "grazing_angle_deg": grazing_angle_deg,
            "diffraction_order": 1,
            "depth_nm": 14.9,
        }
    ]
    runner = BatchSimulationRunner(default_diffraction_order=1, default_fourier_orders=3)
    results = list(runner.run_cases(user_cases))
    csv_path = tmp_path / "batch_user_cases_all_orders.csv"

    write_all_orders_csv(results, csv_path)

    assert len(results) == 1
    assert results[0].status == "ok"
    assert csv_path.exists()


