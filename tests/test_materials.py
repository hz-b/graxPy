from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from grax.gratings import LaminarGrating
from grax.materials import optical_constants_dataframe, resolve_refractive_index, validate_material_input
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

    index = resolve_refractive_index(material, 200.0)

    assert index == pytest.approx(1.0 - 200.0e-6 + 1j * 200.0e-7)


def test_resolve_refractive_index_normalizes_xrt_absorption_sign() -> None:
    material = NegativeImaginaryFakeXrtMaterial()

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

    constants = optical_constants_dataframe(material, [100.0, 200.0])

    assert list(constants.columns) == ["Energy(eV)", "Delta", "Beta"]
    assert np.allclose(constants["Energy(eV)"], [100.0, 200.0])
    assert np.allclose(constants["Delta"], [100.0e-6, 200.0e-6])
    assert np.allclose(constants["Beta"], [100.0e-7, 200.0e-7])


def test_resolve_refractive_index_rejects_string_material_names() -> None:
    with pytest.raises(TypeError, match="Unsupported material input"):
        resolve_refractive_index("Pt", 150.0)


def test_validate_material_input_rejects_string_material_names_with_field_name() -> None:
    with pytest.raises(TypeError, match="substrate_material='Pt'.*cannot be simulated directly"):
        validate_material_input("Pt", field_name="substrate_material")


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

    textures, _ = grating.build_textures(200.0)

    assert textures[0] == 1.0 + 0.0j
    assert textures[-1] == pytest.approx(1.0 - 200.0e-6 + 1j * 200.0e-7)


def test_run_simulation_rejects_string_material_names_early() -> None:
    grating = LaminarGrating(
        substrate_material="Si",
        layer_material="Au",
        layer_thickness_nm=2.0,
        x_resolution_nm=250.0,
        z_resolution_nm=2.0,
    )

    with pytest.raises(TypeError, match="substrate_material='Si'.*xrt Material object"):
        run_simulation(
            grating=grating,
            energy_ev=150.0,
            grazing_angle_deg=4.0,
            diffraction_order=1,
            fourier_orders=3,
        )


def test_runnable_examples_use_local_optical_constants_directories() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    example_scripts = [
        repo_root / "examples" / "laminar_batch_example" / "fixed_angle_sweep.py",
        repo_root / "examples" / "blazed_batch_example" / "fixed_angle_sweep.py",
        repo_root / "examples" / "multilayer_batch_example" / "laminar_multilayer_fixed_angle_sweep.py",
        repo_root / "examples" / "manual_single_simulation_example" / "manual_single_simulation.py",
        repo_root / "elisa_400" / "elisa_400lmm_top_C.py",
    ]

    for script_path in example_scripts:
        script = script_path.read_text(encoding="utf-8")
        assert "pd.read_csv(" in script
        assert (script_path.parent / "optical_constants").is_dir()
        assert "base_dir=" not in script


def test_laminar_xrt_comparison_example_documents_henke_table() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "examples" / "laminar_xrt_comparison" / "fixed_angle_sweep.py"

    script = script_path.read_text(encoding="utf-8")

    assert 'table="Henke"' in script
    assert "Chantler" in script
    assert "BrCo" in script
    assert "pd.read_csv(" in script
    assert "file-backed" in script
    assert "xrt-Henke" in script
