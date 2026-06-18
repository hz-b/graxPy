from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
import warnings

import numpy as np

DATAFRAME_COLUMNS = ("Energy(eV)", "Delta", "Beta")
CLASSICAL_ELECTRON_RADIUS_CM = 2.8179403262e-13
PLANCK_C_EV_CM = 1.2398419843320026e-4
AVOGADRO_NUMBER = 6.02214076e23


@dataclass(frozen=True)
class HenkeOpticalConstants:
    """Parsed Henke optical constants in grax-native form.

    Attributes:
        energy_ev: Photon energies in electronvolts.
        delta: Refractive-index decrement values.
        beta: Absorption values in grax sign convention.
        symbol: Canonical chemical symbol for display and cache keys.
    """

    energy_ev: np.ndarray
    delta: np.ndarray
    beta: np.ndarray
    symbol: str


@dataclass(frozen=True)
class MaterialSpec:
    """Explicit material selection with an optional density override.

    Attributes:
        name: Material name or elemental symbol.
        density_g_cm3: Optional density override in grams per cubic centimeter.
    """

    name: str
    density_g_cm3: float | None = None


ELEMENT_DENSITIES_G_CM3: dict[str, float] = {
    "Ac": 10.07,
    "Ag": 10.501,
    "Al": 2.70,
    "Ar": 0.0017837,
    "As": 5.776,
    "At": 7.0,
    "Au": 19.282,
    "B": 2.37,
    "Ba": 3.62,
    "Be": 1.85,
    "Bi": 9.807,
    "Br": 3.11,
    "C": 2.2670,
    "Ca": 1.54,
    "Cd": 8.69,
    "Ce": 6.770,
    "Cl": 0.003214,
    "Co": 8.86,
    "Cr": 7.15,
    "Cs": 1.93,
    "Cu": 8.933,
    "Dy": 8.55,
    "Er": 9.07,
    "Eu": 5.24,
    "F": 0.001696,
    "Fe": 7.874,
    "Fr": 2.9,
    "Ga": 5.91,
    "Gd": 7.90,
    "Ge": 5.323,
    "H": 0.00008988,
    "He": 0.0001785,
    "Hf": 13.3,
    "Hg": 13.5336,
    "Ho": 8.80,
    "I": 4.93,
    "In": 7.31,
    "Ir": 22.42,
    "K": 0.89,
    "Kr": 0.003733,
    "La": 6.15,
    "Li": 0.534,
    "Lu": 9.84,
    "Mg": 1.74,
    "Mn": 7.3,
    "Mo": 10.2,
    "N": 0.0012506,
    "Na": 0.97,
    "Nb": 8.57,
    "Nd": 7.01,
    "Ne": 0.0008999,
    "Ni": 8.912,
    "O": 0.001429,
    "Os": 22.57,
    "P": 1.82,
    "Pa": 15.37,
    "Pb": 11.342,
    "Pd": 12.0,
    "Pm": 7.26,
    "Po": 9.32,
    "Pr": 6.77,
    "Pt": 21.46,
    "Ra": 5.0,
    "Rb": 1.53,
    "Re": 20.8,
    "Rh": 12.4,
    "Rn": 0.00973,
    "Ru": 12.1,
    "S": 2.067,
    "Sb": 6.685,
    "Sc": 2.99,
    "Se": 4.809,
    "Si": 2.3296,
    "Sm": 7.52,
    "Sn": 7.287,
    "Sr": 2.64,
    "Ta": 16.4,
    "Tb": 8.23,
    "Tc": 11.0,
    "Te": 6.232,
    "Th": 11.72,
    "Ti": 4.5,
    "Tl": 11.8,
    "Tm": 9.32,
    "U": 18.95,
    "V": 6.0,
    "W": 19.3,
    "Xe": 0.005887,
    "Y": 4.47,
    "Yb": 6.90,
    "Zn": 7.134,
    "Zr": 6.52,
}

ELEMENT_ATOMIC_WEIGHTS_G_MOL: dict[str, float] = {
    "Ac": 227.0,
    "Ag": 107.8682,
    "Al": 26.9815384,
    "Ar": 39.95,
    "As": 74.921595,
    "At": 210.0,
    "Au": 196.96657,
    "B": 10.81,
    "Ba": 137.327,
    "Be": 9.0121831,
    "Bi": 208.9804,
    "Br": 79.904,
    "C": 12.011,
    "Ca": 40.078,
    "Cd": 112.414,
    "Ce": 140.116,
    "Cl": 35.45,
    "Co": 58.933194,
    "Cr": 51.9961,
    "Cs": 132.90545196,
    "Cu": 63.546,
    "Dy": 162.5,
    "Er": 167.259,
    "Eu": 151.964,
    "F": 18.998403162,
    "Fe": 55.845,
    "Fr": 223.0,
    "Ga": 69.723,
    "Gd": 157.25,
    "Ge": 72.63,
    "H": 1.008,
    "He": 4.002602,
    "Hf": 178.486,
    "Hg": 200.592,
    "Ho": 164.930329,
    "I": 126.90447,
    "In": 114.818,
    "Ir": 192.217,
    "K": 39.0983,
    "Kr": 83.798,
    "La": 138.90547,
    "Li": 6.94,
    "Lu": 174.9668,
    "Mg": 24.305,
    "Mn": 54.938043,
    "Mo": 95.95,
    "N": 14.007,
    "Na": 22.98976928,
    "Nb": 92.90637,
    "Nd": 144.242,
    "Ne": 20.1797,
    "Ni": 58.6934,
    "O": 15.999,
    "Os": 190.23,
    "P": 30.973761998,
    "Pa": 231.03588,
    "Pb": 207.2,
    "Pd": 106.42,
    "Pm": 145.0,
    "Po": 209.0,
    "Pr": 140.90766,
    "Pt": 195.084,
    "Ra": 226.0,
    "Rb": 85.4678,
    "Re": 186.207,
    "Rh": 102.90549,
    "Rn": 222.0,
    "Ru": 101.07,
    "S": 32.06,
    "Sb": 121.76,
    "Sc": 44.955907,
    "Se": 78.971,
    "Si": 28.085,
    "Sm": 150.36,
    "Sn": 118.71,
    "Sr": 87.62,
    "Ta": 180.94788,
    "Tb": 158.925354,
    "Tc": 98.0,
    "Te": 127.6,
    "Th": 232.0377,
    "Ti": 47.867,
    "Tl": 204.38,
    "Tm": 168.934219,
    "U": 238.02891,
    "V": 50.9415,
    "W": 183.84,
    "Xe": 131.293,
    "Y": 88.905838,
    "Yb": 173.045,
    "Zn": 65.38,
    "Zr": 91.224,
}


def material_label(material: Any) -> str:
    """Return a stable display label for a material input.

    Args:
        material: Material input accepted by :func:`resolve_refractive_index`.

    Returns:
        Human-readable material label for plots and debug output.
    """

    if isinstance(material, MaterialSpec):
        normalized_symbol = _normalize_material_symbol(material.name)
        return normalized_symbol if normalized_symbol is not None else material.name.strip()

    if isinstance(material, str):
        normalized_symbol = _normalize_material_symbol(material)
        return normalized_symbol if normalized_symbol is not None else material.strip()

    name = getattr(material, "name", None)
    if name:
        return str(name)

    if _is_dataframe_like(material):
        dataframe_name = getattr(getattr(material, "attrs", {}), "get", lambda *_: None)("name")
        if dataframe_name:
            return str(dataframe_name)
        return "Optical Constants"

    return material.__class__.__name__


def material_density_g_cm3(material: Any) -> float | None:
    """Return the effective density for a supported material input.

    Args:
        material: Material input accepted by :func:`resolve_refractive_index`.

    Returns:
        Density in grams per cubic centimeter when available, otherwise ``None``.
    """

    if isinstance(material, MaterialSpec):
        if material.density_g_cm3 is not None:
            return float(material.density_g_cm3)
        symbol = _normalize_material_symbol(material.name)
        if symbol is None:
            return None
        return ELEMENT_DENSITIES_G_CM3.get(symbol)

    if isinstance(material, str):
        symbol = _normalize_material_symbol(material)
        if symbol is None:
            return None
        return ELEMENT_DENSITIES_G_CM3.get(symbol)

    density = getattr(material, "density_g_cm3", None)
    if density is not None:
        return float(density)

    rho = getattr(material, "rho", None)
    if rho is not None:
        return float(rho)

    attrs = getattr(material, "attrs", None)
    if isinstance(attrs, dict):
        density_attr = attrs.get("density_g_cm3", attrs.get("rho"))
        if density_attr is not None:
            return float(density_attr)

    return None


def material_density_catalog() -> tuple[tuple[str, float], ...]:
    """Return the built-in elemental Henke density catalog.

    Returns:
        Sorted ``(symbol, density_g_cm3)`` rows used by the runtime string
        material resolver and the documentation/Web UI.
    """

    return tuple(sorted(ELEMENT_DENSITIES_G_CM3.items()))


def available_material_symbols() -> tuple[str, ...]:
    """Return all elemental symbols with packaged Henke tables."""

    return tuple(sorted(_available_henke_symbols()))


def resolve_refractive_index(
    material: Any,
    photon_energy_ev: float,
) -> complex:
    """Resolve any supported material input to a complex refractive index.

    Args:
        material: Material input. Supported values are elemental string
            symbols backed by packaged Henke tables, xrt-like objects with
            ``get_refractive_index()``, and pandas DataFrames with columns
            ``Energy(eV)``, ``Delta``, and ``Beta``.
        photon_energy_ev: Photon energy in electronvolts.

    Returns:
        Complex refractive index at the requested photon energy.

    Raises:
        ValueError: If a DataFrame-like input is missing required columns or
            an xrt-like input returns a non-scalar value for a scalar energy.
        TypeError: If the material input is unsupported.
    """

    if _is_dataframe_like(material):
        return _interpolate_dataframe_index(material, photon_energy_ev)

    if isinstance(material, MaterialSpec):
        return _interpolate_henke_index(
            material.name,
            photon_energy_ev,
            density_g_cm3=material.density_g_cm3,
        )

    if isinstance(material, str):
        return _interpolate_henke_index(material, photon_energy_ev)

    get_refractive_index = getattr(material, "get_refractive_index", None)
    if callable(get_refractive_index):
        warnings.warn(
            "xrt material objects are deprecated and will be removed in a future version. "
            "Use a string material name, MaterialSpec, or a DataFrame-like optical-constants table instead.",
            FutureWarning,
            stacklevel=2,
        )
        refractive_index = np.asarray(get_refractive_index(photon_energy_ev))
        if refractive_index.size != 1:
            raise ValueError("get_refractive_index() must return one value for a scalar energy.")
        scalar_index = complex(refractive_index.reshape(-1)[0])
        return complex(scalar_index.real, abs(scalar_index.imag))

    raise TypeError(
        "Unsupported material input. Expected a string elemental symbol with a packaged Henke table, "
        "a MaterialSpec with an optional density override, a pandas DataFrame-like object with "
        "Energy(eV), Delta, and Beta columns, or an object with get_refractive_index()."
    )


def validate_material_input(
    material: Any,
    *,
    field_name: str | None = None,
) -> None:
    """Validate one material input before simulation-time resolution.

    Args:
        material: Material input to validate. Supported values are elemental
            string symbols backed by packaged Henke tables, DataFrame-like
            optical-constants tables, and objects with
            ``get_refractive_index()``.
        field_name: Optional grating or stack field name used in error messages.

    Raises:
        TypeError: If the material input is unsupported.
    """

    if isinstance(material, MaterialSpec):
        _validate_henke_material_name(
            material.name,
            density_g_cm3=material.density_g_cm3,
            field_name=field_name,
        )
        return

    if isinstance(material, str):
        _validate_henke_material_name(material, field_name=field_name)
        return

    if _is_dataframe_like(material):
        _validate_dataframe_columns(material)
        return

    get_refractive_index = getattr(material, "get_refractive_index", None)
    if callable(get_refractive_index):
        return

    field_prefix = f"{field_name} " if field_name else ""
    raise TypeError(
        f"Unsupported {field_prefix}material input. Expected a pandas DataFrame-like object "
        "with Energy(eV), Delta, and Beta columns or an object with get_refractive_index()."
    )


def optical_constants_dataframe(material: Any, energies_ev: Any) -> Any:
    """Export a material to a DataFrame with grax optical constants.

    Args:
        material: Material input accepted by :func:`resolve_refractive_index`.
        energies_ev: Energies in electronvolts.

    Returns:
        pandas DataFrame with columns ``Energy(eV)``, ``Delta``, and ``Beta``.

    Raises:
        ImportError: If pandas is not installed.
    """

    import pandas as pd

    energies = np.asarray(energies_ev, dtype=float)
    indices = np.asarray([resolve_refractive_index(material, energy) for energy in energies])
    dataframe = pd.DataFrame(
        {
            "Energy(eV)": energies,
            "Delta": 1.0 - np.real(indices),
            "Beta": np.imag(indices),
        }
    )
    material_name = material_label(material)
    if material_name:
        dataframe.attrs["name"] = str(material_name)
    return dataframe


def _normalize_material_symbol(material_name: str) -> str | None:
    """Normalize a material string to an elemental symbol when possible.

    Args:
        material_name: User-provided material name.

    Returns:
        Canonical chemical symbol when the input looks like an elemental symbol,
        otherwise ``None``.
    """

    stripped = material_name.strip()
    if not stripped.isalpha() or len(stripped) > 2:
        return None
    if len(stripped) == 1:
        return stripped.upper()
    return stripped[0].upper() + stripped[1:].lower()


def _validate_henke_material_name(
    material_name: str,
    *,
    density_g_cm3: float | None = None,
    field_name: str | None = None,
) -> None:
    """Validate that a material string can be resolved through packaged Henke data.

    Args:
        material_name: User-provided material name.
        field_name: Optional caller field name for targeted error messages.

    Raises:
        ValueError: If the string is not a supported elemental Henke material.
    """

    symbol = _normalize_material_symbol(material_name)
    if symbol is None:
        raise ValueError(
            f"Material {material_name!r} is not available. String materials must be elemental symbols "
            "with packaged Henke tables, or use a MaterialSpec, a DataFrame-like optical-constants table, "
            "or an object with get_refractive_index(). Available materials: "
            f"{_available_henke_error_list()}"
        )
    if symbol not in _available_henke_symbols():
        raise ValueError(
            f"Material {symbol!r} is not available in the packaged Henke tables. Available materials: "
            f"{_available_henke_error_list()}"
        )
    if density_g_cm3 is not None:
        if float(density_g_cm3) <= 0.0:
            raise ValueError("density_g_cm3 must be positive.")
        return
    if symbol not in ELEMENT_DENSITIES_G_CM3:
        raise ValueError(
            f"Material {symbol!r} is available, but no density metadata is configured yet. "
            "Provide density_g_cm3 explicitly or add the element to the built-in density registry."
        )


@lru_cache(maxsize=None)
def _available_henke_symbols() -> frozenset[str]:
    """Return the set of elemental symbols with packaged Henke tables."""

    table_root = Path(__file__).resolve().parent / "henke_tables"
    symbols = set()
    for table_path in table_root.glob("*.nff"):
        symbols.add(_normalize_material_symbol(table_path.stem))
    return frozenset(symbol for symbol in symbols if symbol is not None)


@lru_cache(maxsize=None)
def _available_henke_error_list() -> str:
    """Return a cached comma-separated list of available Henke symbols."""

    return ", ".join(available_material_symbols())


@lru_cache(maxsize=None)
def _load_henke_optical_constants(
    symbol: str,
    density_g_cm3: float | None = None,
) -> HenkeOpticalConstants:
    """Load one packaged Henke table and convert it to grax optical constants.

    Args:
        symbol: Canonical elemental symbol.

    Returns:
        Parsed Henke optical constants converted to ``delta`` and ``beta``.

    Raises:
        ValueError: If the symbol is unsupported or required metadata is missing.
    """

    if symbol not in _available_henke_symbols():
        raise ValueError(f"No packaged Henke table is available for {symbol!r}.")
    atomic_weight_g_mol = ELEMENT_ATOMIC_WEIGHTS_G_MOL.get(symbol)
    if atomic_weight_g_mol is None:
        raise ValueError(
            f"A packaged Henke table exists for {symbol!r}, but no atomic-weight metadata is configured."
        )

    table_path = Path(__file__).resolve().parent / "henke_tables" / f"{symbol.lower()}.nff"
    energy_ev_values: list[float] = []
    f1_values: list[float] = []
    f2_values: list[float] = []
    with table_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or line.lower().startswith("e(ev)"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            energy_ev = float(parts[0])
            f1 = float(parts[1])
            f2 = float(parts[2])
            if f1 <= -9998.0:
                continue
            energy_ev_values.append(energy_ev)
            f1_values.append(f1)
            f2_values.append(f2)

    if not energy_ev_values:
        raise ValueError(f"Packaged Henke table for {symbol!r} did not contain valid f1/f2 rows.")

    energy_ev = np.asarray(energy_ev_values, dtype=float)
    f1 = np.asarray(f1_values, dtype=float)
    f2 = np.asarray(f2_values, dtype=float)
    effective_density = density_g_cm3 if density_g_cm3 is not None else ELEMENT_DENSITIES_G_CM3.get(symbol)
    if effective_density is None:
        raise ValueError(
            f"A packaged Henke table exists for {symbol!r}, but no density metadata is configured. "
            "Provide density_g_cm3 explicitly or add the element to the built-in density registry."
        )
    number_density_cm3 = effective_density * AVOGADRO_NUMBER / atomic_weight_g_mol
    wavelength_cm = PLANCK_C_EV_CM / energy_ev
    coefficient = number_density_cm3 * CLASSICAL_ELECTRON_RADIUS_CM * (wavelength_cm ** 2) / (
        2.0 * np.pi
    )
    delta = coefficient * f1
    beta = coefficient * f2
    return HenkeOpticalConstants(
        energy_ev=energy_ev,
        delta=np.asarray(delta, dtype=float),
        beta=np.asarray(beta, dtype=float),
        symbol=symbol,
    )


def _interpolate_henke_index(
    material_name: str,
    photon_energy_ev: float,
    *,
    density_g_cm3: float | None = None,
) -> complex:
    """Interpolate one packaged Henke material at the requested energy.

    Args:
        material_name: User-provided elemental material string.
        photon_energy_ev: Photon energy in electronvolts.

    Returns:
        Complex refractive index in grax sign convention.
    """

    _validate_henke_material_name(material_name, density_g_cm3=density_g_cm3)
    symbol = _normalize_material_symbol(material_name)
    assert symbol is not None
    constants = _load_henke_optical_constants(symbol, density_g_cm3)
    return complex(
        1.0 - np.interp(photon_energy_ev, constants.energy_ev, constants.delta)
        + 1j * np.interp(photon_energy_ev, constants.energy_ev, constants.beta)
    )


def _is_dataframe_like(material: Any) -> bool:
    """Return whether material looks like a pandas DataFrame."""

    return hasattr(material, "columns") and hasattr(material, "__getitem__")


def _validate_dataframe_columns(material: Any) -> None:
    """Validate that a DataFrame-like material provides required columns."""

    provided_columns = [str(column) for column in material.columns]
    missing_columns = [column for column in DATAFRAME_COLUMNS if column not in material.columns]
    if missing_columns:
        raise ValueError(
            "Optical constants DataFrame must contain columns: "
            f"{', '.join(DATAFRAME_COLUMNS)}. "
            f"Missing: {', '.join(missing_columns)}. "
            f"Provided: {', '.join(provided_columns)}."
        )


def _interpolate_dataframe_index(material: Any, photon_energy_ev: float) -> complex:
    """Interpolate delta and beta from a DataFrame-like object."""

    _validate_dataframe_columns(material)

    energy = np.asarray(material["Energy(eV)"], dtype=float)
    delta = np.asarray(material["Delta"], dtype=float)
    beta = np.asarray(material["Beta"], dtype=float)
    return complex(
        1.0 - np.interp(photon_energy_ev, energy, delta)
        + 1j * np.interp(photon_energy_ev, energy, beta)
    )


__all__ = [
    "MaterialSpec",
    "available_material_symbols",
    "material_density_catalog",
    "material_density_g_cm3",
    "material_label",
    "optical_constants_dataframe",
    "resolve_refractive_index",
    "validate_material_input",
]
