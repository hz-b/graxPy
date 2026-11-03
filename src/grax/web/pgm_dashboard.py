"""Files and page data of the live SX700 PGM dashboard."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from matplotlib import colormaps

from grax.pgm import SX700Parameters, geometry_for_state, side_view_geometry
from grax.pgm.sampling import _interpolate_rect_grid

from .common import _file_token
from .files import remove_file, replace_file

MAP_KEYS = ("geometry", "mirror", "grating", "optical", "total")
FINAL_FILE = "maps.npz"
BASE_FILE = "live_base.npz"
NODES_FILE = "live_nodes.npz"
OVERLAY_FIELDS = (
    "optical",
    "theta_ok",
    "beta_ok",
    "theta_deg",
    "beta_abs_deg",
    "mirror_grating_distance_mm",
)
POINT_GEOMETRY_FIELDS = (
    "valid",
    "alpha_deg",
    "beta_deg",
    "theta_deg",
    "theta_grazing_deg",
    "mirror_ok",
    "grating_ok",
    "sequence_ok",
    "theta_ok",
    "beta_ok",
    "throughput_percent",
    "mirror_footprint_mm",
    "grating_footprint_mm",
)
LIVE_POINTS = 170
OVERLAY_POINTS = 200
HATCH_POINTS = 120
HATCH_SPACING = 5
TURBO_STOPS = 64


def _write_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write arrays atomically so a page never reads half a file."""
    temporary = path.with_name(f"{path.stem}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        replace_file(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_arrays(path: Path) -> dict[str, np.ndarray] | None:
    """Return a saved array file, or None while it is missing or being replaced."""
    try:
        with np.load(path, allow_pickle=False) as saved:
            return {key: saved[key] for key in saved.files}
    except OSError:
        return None


def _base_arrays(energies, cff, geometry, mirror_percent) -> dict[str, np.ndarray]:
    return {
        "energies": np.asarray(energies, dtype=float),
        "cff": np.asarray(cff, dtype=float),
        "geometry": np.asarray(geometry["throughput_percent"], dtype=float),
        "mirror": np.asarray(mirror_percent, dtype=float),
        **{name: np.asarray(geometry[name]) for name in OVERLAY_FIELDS},
    }


def save_base_maps(run_dir: Path, energies, cff, geometry, mirror_reflectivity) -> None:
    """Save what is known before the first grating node: geometry, mirror and limits."""
    arrays = _base_arrays(energies, cff, geometry, 100.0 * np.asarray(mirror_reflectivity))
    _write_arrays(run_dir / BASE_FILE, arrays)


def save_nodes(run_dir: Path, e_nodes, c_nodes, eta_nodes) -> None:
    """Save the grating nodes solved so far; unsolved nodes are NaN."""
    _write_arrays(
        run_dir / NODES_FILE,
        {"e_nodes": np.asarray(e_nodes), "c_nodes": np.asarray(c_nodes), "eta": eta_nodes},
    )


def save_final_maps(run_dir: Path, energies, cff, geometry, optics) -> None:
    """Save a finished run's maps and drop its live files."""
    arrays = _base_arrays(energies, cff, geometry, 100.0 * optics["mirror_reflectivity"])
    arrays.update(
        grating=100.0 * np.asarray(optics["grating_efficiency"]),
        optical_map=np.asarray(optics["optical_percent"]),
        total=np.asarray(optics["total_percent"]),
    )
    _write_arrays(run_dir / FINAL_FILE, arrays)
    remove_file(run_dir / BASE_FILE)
    remove_file(run_dir / NODES_FILE)


def _with_grating_maps(base: dict[str, np.ndarray], nodes: dict[str, np.ndarray] | None):
    """Add the grating and product maps; areas without solved nodes stay NaN."""
    shape = base["geometry"].shape
    if nodes is None:
        eta = np.full(shape, np.nan)
    else:
        eta = _interpolate_rect_grid(
            nodes["e_nodes"], nodes["c_nodes"], nodes["eta"], base["energies"], base["cff"]
        )
    optical = base.get("optical")
    grating = 100.0 * (np.where(optical, eta, 0.0) if optical is not None else eta)
    product = np.clip(base["mirror"] * grating / 100.0, 0.0, 100.0)
    total = np.clip(base["geometry"] * product / 100.0, 0.0, 100.0)
    return {**base, "grating": grating, "optical_map": product, "total": total}


def load_maps(run_dir: Path) -> dict[str, np.ndarray] | None:
    """Return the finished maps, or the current ones of a running or stopped run."""
    final = _read_arrays(run_dir / FINAL_FILE)
    if final is not None:
        if "optical_map" not in final and "optical" in final:
            # Runs saved before the dashboard kept the product map as "optical".
            final["optical_map"] = final.pop("optical")
        return final
    base = _read_arrays(run_dir / BASE_FILE)
    if base is None:
        return None
    return _with_grating_maps(base, _read_arrays(run_dir / NODES_FILE))


def _token(run_dir: Path) -> str:
    return _file_token(run_dir / FINAL_FILE) or _file_token(run_dir / NODES_FILE)


def _grid(values: np.ndarray, decimals: int = 3) -> list[list[float | None]]:
    """Return a grid for JSON, with None for the NaN values JSON does not allow."""
    rounded = np.round(np.asarray(values, dtype=float), decimals).astype(object)
    rounded[~np.isfinite(np.asarray(values, dtype=float))] = None
    return rounded.tolist()


def _stride(count: int, points: int) -> int:
    return max(1, math.ceil(count / points))


def _nearest(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def _start_point(energies: np.ndarray, cff: np.ndarray) -> dict[str, int]:
    """Return the colleague's example point (1000 eV, CFF 2.25), or mid-range outside the scan."""
    energy = 1000.0 if energies[0] <= 1000.0 <= energies[-1] else 0.5 * (energies[0] + energies[-1])
    factor = 2.25 if cff[0] <= 2.25 <= cff[-1] else 0.5 * (cff[0] + cff[-1])
    return {"energy_index": _nearest(energies, energy), "cff_index": _nearest(cff, factor)}


def _level_in_range(values: np.ndarray, level: float) -> bool:
    finite = values[np.isfinite(values)]
    return finite.size > 0 and float(finite.min()) <= level <= float(finite.max())


def _overlays(maps: dict[str, np.ndarray], values: dict[str, Any]) -> dict[str, Any] | None:
    """Return the mechanical limits, critical area and angle cut for the map overlays."""
    if not all(name in maps for name in OVERLAY_FIELDS):
        return None
    energies, cff = maps["energies"], maps["cff"]
    theta, beta = maps["theta_deg"].astype(float), maps["beta_abs_deg"].astype(float)
    distance = maps["mirror_grating_distance_mm"].astype(float)
    optical = maps["optical"].astype(bool)
    critical_distance = float(values["crit_distance"])

    e_step = _stride(len(energies), OVERLAY_POINTS)
    c_step = _stride(len(cff), OVERLAY_POINTS)

    def thin(grid: np.ndarray) -> np.ndarray:
        return grid[::c_step, ::e_step]

    limits = []
    for field, grid, level, label, dash in (
        ("theta_deg", theta, values["theta_min_deg"], "Θ<sub>min</sub>", "dash"),
        ("theta_deg", theta, values["theta_max_deg"], "Θ<sub>max</sub>", "solid"),
        ("beta_abs_deg", beta, values["beta_min_deg"], "|β|<sub>min</sub>", "dot"),
        ("beta_abs_deg", beta, values["beta_max_deg"], "|β|<sub>max</sub>", "dashdot"),
    ):
        if _level_in_range(grid, float(level)):
            limits.append(
                {
                    "field": field,
                    "level": float(level),
                    "label": f"{label} = {float(level):g}°",
                    "dash": dash,
                }
            )

    critical = optical & np.isfinite(distance) & (distance < critical_distance)
    hatch_e = _stride(len(energies), HATCH_POINTS)
    hatch_c = _stride(len(cff), HATCH_POINTS)
    hatch = critical[::hatch_c, ::hatch_e].astype(float)
    rows, columns = np.indices(hatch.shape)
    hatch[((rows + columns) % HATCH_SPACING != 0) | (hatch == 0)] = np.nan
    cut = thin(optical & ~(maps["theta_ok"].astype(bool) & maps["beta_ok"].astype(bool))).astype(
        float
    )
    cut[cut == 0] = np.nan
    return {
        "energies": energies[::e_step].tolist(),
        "cff": cff[::c_step].tolist(),
        "fields": {"theta_deg": _grid(thin(theta), 2), "beta_abs_deg": _grid(thin(beta), 2)},
        "limits": limits,
        "distance": _grid(thin(distance), 2),
        "distance_limit": {
            "level": critical_distance,
            "label": f"d<sub>M–G</sub> = {critical_distance:g} mm",
            "drawn": _level_in_range(distance, critical_distance),
        },
        "angle_cut": _grid(cut, 0) if np.isfinite(cut).any() else None,
        "hatch": {
            "energies": energies[::hatch_e].tolist(),
            "cff": cff[::hatch_c].tolist(),
            "z": _grid(hatch, 0),
        }
        if critical.any()
        else None,
    }


def _turbo_colorscale() -> list[list[float | str]]:
    """Return matplotlib's turbo colormap as Plotly color stops."""
    turbo = colormaps["turbo"]
    stops = []
    for position in np.linspace(0.0, 1.0, TURBO_STOPS):
        red, green, blue, _ = turbo(float(position), bytes=True)
        stops.append([float(position), f"rgb({red},{green},{blue})"])
    return stops


def dashboard_payload(run_dir: Path, values: dict[str, Any]) -> dict[str, Any] | None:
    """Return all maps and overlays at full resolution, or None before geometry is known."""
    maps = load_maps(run_dir)
    if maps is None:
        return None
    energies, cff = maps["energies"], maps["cff"]
    return {
        "energies": energies.tolist(),
        "cff": cff.tolist(),
        "maps": {
            "geometry": _grid(maps["geometry"]),
            "mirror": _grid(maps["mirror"]),
            "grating": _grid(maps["grating"]),
            "optical": _grid(maps["optical_map"]),
            "total": _grid(maps["total"]),
        },
        "overlays": _overlays(maps, values),
        "colorscale": _turbo_colorscale(),
        "start": _start_point(energies, cff),
        "token": _token(run_dir),
    }


def live_payload(run_dir: Path, version: str) -> dict[str, Any]:
    """Return the three changing maps, thinned, or a note that nothing has changed."""
    token = _token(run_dir)
    if token == version:
        return {"unchanged": True}
    maps = load_maps(run_dir)
    if maps is None:
        return {"unchanged": True}
    e_step = _stride(len(maps["energies"]), LIVE_POINTS)
    c_step = _stride(len(maps["cff"]), LIVE_POINTS)

    def thin(name: str) -> list[list[float | None]]:
        return _grid(maps[name][::c_step, ::e_step])

    return {
        "token": token,
        "energies": maps["energies"][::e_step].tolist(),
        "cff": maps["cff"][::c_step].tolist(),
        "maps": {
            "grating": thin("grating"),
            "optical": thin("optical_map"),
            "total": thin("total"),
        },
    }


def point_efficiencies(maps: dict[str, np.ndarray], cff_index: int, energy_index: int):
    """Return one map point's efficiencies; a value not solved yet is None."""

    def value(name: str) -> float | None:
        number = float(maps[name][cff_index, energy_index])
        return number if math.isfinite(number) else None

    return {
        "mirror": value("mirror"),
        "grating": value("grating"),
        "optical": value("optical_map"),
        "total": value("total"),
    }


def point_detail(
    maps: dict[str, np.ndarray], energy_index: int, cff_index: int, parameters: SX700Parameters
) -> dict[str, Any]:
    """Return the geometry, side view and efficiencies of one map point."""
    energy = float(maps["energies"][energy_index])
    cff = float(maps["cff"][cff_index])
    geometry = geometry_for_state(energy, cff, parameters)
    return {
        "energy_eV": energy,
        "cff": cff,
        "energy_index": energy_index,
        "cff_index": cff_index,
        "geometry": (
            None if geometry is None else {key: geometry[key] for key in POINT_GEOMETRY_FIELDS}
        ),
        "side_view": side_view_geometry(energy, cff, parameters),
        "efficiencies": point_efficiencies(maps, cff_index, energy_index),
    }
