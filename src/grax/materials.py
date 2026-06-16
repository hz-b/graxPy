from __future__ import annotations

from typing import Any

import numpy as np

DATAFRAME_COLUMNS = ("Energy(eV)", "Delta", "Beta")


def material_label(material: Any) -> str:
    """Return a stable display label for a material input.

    Args:
        material: Material input accepted by :func:`resolve_refractive_index`.

    Returns:
        Human-readable material label for plots and debug output.
    """

    if isinstance(material, str):
        return material

    name = getattr(material, "name", None)
    if name:
        return str(name)

    if _is_dataframe_like(material):
        dataframe_name = getattr(getattr(material, "attrs", {}), "get", lambda *_: None)("name")
        if dataframe_name:
            return str(dataframe_name)
        return "Optical Constants"

    return material.__class__.__name__


def resolve_refractive_index(
    material: Any,
    photon_energy_ev: float,
) -> complex:
    """Resolve any supported material input to a complex refractive index.

    Args:
        material: Material input. Supported values are xrt-like objects with
            ``get_refractive_index()`` and pandas DataFrames with columns
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

    get_refractive_index = getattr(material, "get_refractive_index", None)
    if callable(get_refractive_index):
        refractive_index = np.asarray(get_refractive_index(photon_energy_ev))
        if refractive_index.size != 1:
            raise ValueError("get_refractive_index() must return one value for a scalar energy.")
        scalar_index = complex(refractive_index.reshape(-1)[0])
        return complex(scalar_index.real, abs(scalar_index.imag))

    raise TypeError(
        "Unsupported material input. Expected a pandas DataFrame or object with "
        "get_refractive_index()."
    )


def validate_material_input(
    material: Any,
    *,
    field_name: str | None = None,
) -> None:
    """Validate one material input before simulation-time resolution.

    Args:
        material: Material input to validate.
        field_name: Optional grating or stack field name used in error messages.

    Raises:
        TypeError: If the material input is unsupported.
    """

    if isinstance(material, str):
        field_prefix = f"{field_name}=" if field_name else "material="
        raise TypeError(
            f"Unsupported {field_prefix}{material!r}. Bare string material names are labels only "
            "and cannot be simulated directly. Pass an xrt Material object or a "
            "DataFrame-like optical-constants table with Energy(eV), Delta, and Beta columns."
        )

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
    material_name = getattr(material, "name", None)
    if material_name:
        dataframe.attrs["name"] = str(material_name)
    return dataframe


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
