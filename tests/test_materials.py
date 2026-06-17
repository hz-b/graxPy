from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from xrt.backends.raycing import materials as xrt_materials

from grax.gratings import LaminarGrating
from grax.materials import MaterialSpec, optical_constants_dataframe, resolve_refractive_index, validate_material_input
from grax.simulation import run_simulation
from tests.optical_constants import OpticalConstantsTable


class FakeXrtMaterial:
    """Minimal xrt-like material used by material adapter tests."""

    name = "FakeXrt"

    def get_refractive_index(self, energy_ev: float | np.ndarray) -> complex | np.ndarray:
        """Return an energy-dependent complex refractive index."""

        energy = np.asarray(energy_ev, dtype=float)
        index = 1.0 - energy * 1e-6 + 1j * energy * 1e-7
        if index.ndim == 0:
            return complex(index)
        return index


class NegativeImaginaryFakeXrtMaterial:
    """xrt-like material using xrt's absorbing-index sign convention."""

    name = "NegativeImaginaryFakeXrt"

    def get_refractive_index(self, energy_ev: float) -> complex:
        """Return a complex index with negative imaginary absorption."""

        return 1.0 - energy_ev * 1e-6 - 1j * energy_ev * 1e-7


def test_resolve_refractive_index_accepts_xrt_like_material() -> None:
    material = FakeXrtMaterial()

    with pytest.warns(FutureWarning, match="deprecated"):
        index = resolve_refractive_index(material, 200.0)

    assert index == pytest.approx(1.0 - 200.0e-6 + 1j * 200.0e-7)


def test_resolve_refractive_index_normalizes_xrt_absorption_sign() -> None:
    material = NegativeImaginaryFakeXrtMaterial()

    with pytest.warns(FutureWarning, match="deprecated"):
        index = resolve_refractive_index(material, 200.0)

    assert index == pytest.approx(1.0 - 200.0e-6 + 1j * 200.0e-7)


def test_resolve_refractive_index_accepts_dataframe_delta_beta() -> None:
    pd = pytest.importorskip("pandas")
    constants = pd.DataFrame(
        {
            "Energy(eV)": [100.0, 200.0, 300.0],
            "Delta": [1.0e-4, 2.0e-4, 3.0e-4],
            "Beta": [1.0e-5, 2.0e-5, 3.0e-5],
        }
    )

    index = resolve_refractive_index(constants, 150.0)

    assert index == pytest.approx(1.0 - 1.5e-4 + 1j * 1.5e-5)


def test_resolve_refractive_index_rejects_dataframe_without_required_columns() -> None:
    pd = pytest.importorskip("pandas")
    constants = pd.DataFrame({"Energy": [100.0], "Delta": [1e-4], "Beta": [1e-5]})

    with pytest.raises(ValueError, match="Missing: Energy\\(eV\\).*Provided: Energy, Delta, Beta"):
        resolve_refractive_index(constants, 100.0)


@pytest.mark.parametrize(
    ("columns", "missing_columns"),
    [
        ({"Energy": [100.0], "Delta": [1e-4], "Beta": [1e-5]}, ["Energy(eV)"]),
        ({"Energy(eV)": [100.0], "Deltas": [1e-4], "Beta": [1e-5]}, ["Delta"]),
        ({"Energy(eV)": [100.0], "Delta": [1e-4], "beta": [1e-5]}, ["Beta"]),
        ({"energy_ev": [100.0], "Delta ": [1e-4], "beta": [1e-5]}, ["Energy(eV)", "Delta", "Beta"]),
    ],
)
def test_resolve_refractive_index_rejects_invalid_dataframe_columns(
    columns: dict[str, list[float]],
    missing_columns: list[str],
) -> None:
    pd = pytest.importorskip("pandas")
    constants = pd.DataFrame(columns)

    with pytest.raises(ValueError) as error_info:
        resolve_refractive_index(constants, 100.0)

    error_message = str(error_info.value)
    for column in missing_columns:
        assert column in error_message
    assert "Provided:" in error_message


def test_resolve_refractive_index_accepts_dataframe_with_extra_columns() -> None:
    pd = pytest.importorskip("pandas")
    constants = pd.DataFrame(
        {
            "Energy(eV)": [100.0, 200.0],
            "Delta": [1.0e-4, 2.0e-4],
            "Beta": [1.0e-5, 2.0e-5],
            "Notes": ["a", "b"],
        }
    )

    index = resolve_refractive_index(constants, 150.0)

    assert index == pytest.approx(1.0 - 1.5e-4 + 1j * 1.5e-5)


def test_grating_build_textures_rejects_dataframe_with_invalid_material_columns() -> None:
    pd = pytest.importorskip("pandas")
    invalid_constants = pd.DataFrame({"Energy": [100.0], "Delta": [1e-4], "Beta": [1e-5]})
    valid_constants = pd.DataFrame(
        {
            "Energy(eV)": [100.0, 200.0],
            "Delta": [1.0e-4, 2.0e-4],
            "Beta": [1.0e-5, 2.0e-5],
        }
    )
    grating = LaminarGrating(
        substrate_material=invalid_constants,
        layer_material=valid_constants,
        layer_thickness_nm=2.0,
        x_resolution_nm=250.0,
        z_resolution_nm=2.0,
    )

    with pytest.raises(ValueError, match="Missing: Energy\\(eV\\)"):
        grating.build_textures(150.0)


def test_optical_constants_dataframe_exports_delta_beta_columns() -> None:
    pytest.importorskip("pandas")
    material = FakeXrtMaterial()

    with pytest.warns(FutureWarning, match="deprecated"):
        constants = optical_constants_dataframe(material, [100.0, 200.0])

    assert list(constants.columns) == ["Energy(eV)", "Delta", "Beta"]
    assert np.allclose(constants["Energy(eV)"], [100.0, 200.0])
    assert np.allclose(constants["Delta"], [100.0e-6, 200.0e-6])
    assert np.allclose(constants["Beta"], [100.0e-7, 200.0e-7])


def test_resolve_refractive_index_accepts_henke_string_material_names() -> None:
    index = resolve_refractive_index("Pt", 150.0)

    assert np.isfinite(index.real)
    assert np.isfinite(index.imag)
    assert index.imag >= 0.0


def test_validate_material_input_accepts_henke_string_material_names() -> None:
    validate_material_input("Pt", field_name="substrate_material")


def test_material_spec_density_override_resolves_and_matches_xrt() -> None:
    material = MaterialSpec("Ag", density_g_cm3=10.49)
    xrt_material = xrt_materials.Material("Ag", rho=10.49, table="Henke", name="Ag-xrt")

    index = resolve_refractive_index(material, 150.0)
    with pytest.warns(FutureWarning, match="deprecated"):
        xrt_index = resolve_refractive_index(xrt_material, 150.0)

    assert index == pytest.approx(xrt_index, rel=2e-2, abs=2e-6)


def test_resolve_refractive_index_accepts_case_insensitive_henke_string_material_names() -> None:
    upper = resolve_refractive_index("Pt", 150.0)
    lower = resolve_refractive_index("pt", 150.0)

    assert lower == pytest.approx(upper)


def test_resolve_refractive_index_rejects_unknown_string_material_names() -> None:
    with pytest.raises(ValueError) as error_info:
        resolve_refractive_index("Xx", 150.0)

    message = str(error_info.value)
    assert "not available" in message
    assert "Available materials:" in message
    assert "Si" in message
    assert "Pt" in message


def test_validate_material_input_rejects_henke_materials_without_density_metadata() -> None:
    with pytest.raises(ValueError, match="no density metadata is configured"):
        validate_material_input("Ag", field_name="substrate_material")


def test_resolve_refractive_index_rejects_numeric_constants() -> None:
    with pytest.raises(TypeError, match="Unsupported material input"):
        resolve_refractive_index(1.0 - 1.5e-4 + 1j * 1.5e-5, 150.0)


def test_grating_build_textures_accepts_xrt_like_materials() -> None:
    material = FakeXrtMaterial()
    coating = OpticalConstantsTable(
        energy_ev=np.asarray([100.0, 200.0, 300.0], dtype=float),
        delta=np.asarray([100.0e-6, 200.0e-6, 300.0e-6], dtype=float),
        beta=np.asarray([100.0e-7, 200.0e-7, 300.0e-7], dtype=float),
        name="Fake coating",
    )
    grating = LaminarGrating(
        substrate_material=material,
        layer_material=coating,
        layer_thickness_nm=2.0,
        x_resolution_nm=250.0,
        z_resolution_nm=2.0,
    )

    with pytest.warns(FutureWarning, match="deprecated"):
        textures, _ = grating.build_textures(200.0)

    assert textures[0] == 1.0 + 0.0j
    assert textures[-1] == pytest.approx(1.0 - 200.0e-6 + 1j * 200.0e-7)


def test_run_simulation_accepts_henke_string_material_names() -> None:
    grating = LaminarGrating(
        substrate_material="Si",
        layer_material="Pt",
        layer_thickness_nm=2.0,
        x_resolution_nm=250.0,
        z_resolution_nm=2.0,
    )

    result = run_simulation(
        grating=grating,
        energy_ev=150.0,
        grazing_angle_deg=4.0,
        diffraction_order=1,
        fourier_orders=3,
    )

    assert result.selected_efficiency >= 0.0


def test_run_simulation_rejects_unknown_string_material_names_early() -> None:
    grating = LaminarGrating(
        substrate_material="Si",
        layer_material="Xx",
        layer_thickness_nm=2.0,
        x_resolution_nm=250.0,
        z_resolution_nm=2.0,
    )

    with pytest.raises(ValueError, match="not available"):
        run_simulation(
            grating=grating,
            energy_ev=150.0,
            grazing_angle_deg=4.0,
            diffraction_order=1,
            fourier_orders=3,
        )


def test_optical_constants_dataframe_exports_henke_string_material() -> None:
    pytest.importorskip("pandas")

    constants = optical_constants_dataframe("Pt", [100.0, 200.0])

    assert list(constants.columns) == ["Energy(eV)", "Delta", "Beta"]
    assert list(constants["Energy(eV)"]) == [100.0, 200.0]
    assert np.all(constants["Delta"] >= 0.0)
    assert np.all(constants["Beta"] >= 0.0)
    assert constants.attrs["name"] == "Pt"


@pytest.mark.parametrize(
    ("symbol", "density_g_cm3"),
    [
        ("Si", 2.329),
        ("Pt", 21.45),
        ("Au", 19.3),
        ("C", 2.2),
        ("Cr", 7.19),
    ],
)
def test_henke_string_resolution_matches_xrt_henke_close_enough(
    symbol: str,
    density_g_cm3: float,
) -> None:
    xrt_material = xrt_materials.Material(symbol, rho=density_g_cm3, table="Henke", name=f"{symbol}-xrt")

    for energy_ev in [100.0, 200.0, 1000.0]:
        henke_index = resolve_refractive_index(symbol, energy_ev)
        xrt_index = resolve_refractive_index(xrt_material, energy_ev)
        assert henke_index == pytest.approx(xrt_index, rel=2e-2, abs=2e-6)


def test_packaged_henke_tables_are_discoverable() -> None:
    import grax.materials as materials_module

    table_path = Path(materials_module.__file__).resolve().parent / "henke_tables" / "si.nff"

    assert table_path.is_file()


def test_runnable_examples_use_local_optical_constants_directories() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    example_scripts = [
        repo_root / "examples" / "simulation" / "multilayer_theta_search" / "multilayer_theta_search.py",
        repo_root / "examples" / "optimizer" / "optimizer_laminar" / "0_fit_laminar_grating.py",
        repo_root / "examples" / "optimizer" / "optimizer_blazed" / "1_run_simulation_design_parameters.py",
    ]

    for script_path in example_scripts:
        script = script_path.read_text(encoding="utf-8")
        assert "pd.read_csv(" in script
        assert (script_path.parent / "optical_constants").is_dir()
        assert "optical_constants_dir" in script


def test_materials_tutorial_documents_string_and_xrt_material_paths() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "docs" / "tutorials" / "materials.md"

    script = script_path.read_text(encoding="utf-8")

    assert 'substrate_material="Si"' in script
    assert 'layer_material="Pt"' in script
    assert 'table="Henke"' in script
    assert "get_refractive_index()" in script
    assert "DataFrame-like object" in script
