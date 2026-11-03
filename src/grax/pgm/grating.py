# ruff: noqa: D403, D212, N803, E501
# Preserve the colleague's scientific notation and descriptions.
"""Grating efficiency from graxpy: angle/order mapping, batch evaluation, description."""

import hashlib
import warnings

import numpy as np

import grax
from grax.simulation import _resolve_max_workers

from .geometry import solve_alpha_beta
from .parameters import SX700Parameters

# =====================================================================
# GRATING EFFICIENCY (graxpy)
# =====================================================================
#
# The grating is a graxpy object (grax.BlazedGrating, grax.LaminarGrating,
# ...) passed as SX700Parameters(grating=...). graxpy measures the incidence
# angle from the grating SURFACE (grazing) and selects, for a positive
# diffraction order m, the order that satisfies
#     sin(alpha) + sin(beta) = m * lambda * N     with beta < 0,
# which is the convention of this script (alpha = 90 deg - grazing angle).

_Z_RESOLUTION_WARNED = set()


def _require_grating(p):
    if p.grating is None:
        raise ValueError(
            "SX700Parameters.grating is required for efficiency calculations. "
            "Build one with graxpy, e.g. grax.BlazedGrating(...), and pass "
            "SX700Parameters(grating=...)."
        )
    return p.grating


def _require_s_polarization(p):
    if str(p.polarization).strip().lower() not in ("s", "te"):
        raise NotImplementedError("The mirror and grating model currently supports s (TE) polarization only.")


def _grating_stack(grating):
    try:
        return grating.resolved_stack()
    except Exception:
        return None


def _grating_layers_bottom_up(grating):
    """[(material, thickness_nm), ...] of the coating stack, bottom to top."""
    stack = _grating_stack(grating)
    if stack is None:
        return []
    try:
        return list(stack.layer_sequence_bottom_up())
    except Exception:
        return []


def _grating_material_names(grating):
    stack = _grating_stack(grating)
    if stack is None:
        return []
    try:
        return [str(name) for name in stack.material_names()]
    except Exception:
        return []


def _material_label_text(material):
    name = getattr(material, "name", None)
    if isinstance(name, str):
        return name
    if isinstance(material, str):
        return material
    return type(material).__name__


def _warn_if_z_resolution_quantises(grating):
    """graxpy places layer boundaries on its z grid; warn if thicknesses are rounded."""
    z_res = float(getattr(grating, "z_resolution_nm", 0.0) or 0.0)
    if z_res <= 0.0 or id(grating) in _Z_RESOLUTION_WARNED:
        return
    off_grid = []
    for material, thickness in _grating_layers_bottom_up(grating):
        ratio = float(thickness) / z_res
        if abs(ratio - round(ratio)) > 1e-3:
            off_grid.append(f"{_material_label_text(material)} {float(thickness):g} nm")
    if off_grid:
        _Z_RESOLUTION_WARNED.add(id(grating))
        warnings.warn(
            f"Layer thicknesses are not multiples of the grating z_resolution_nm={z_res:g}: "
            + ", ".join(sorted(set(off_grid)))
            + ". graxpy rounds layer boundaries to its z grid; choose a z_resolution_nm "
            "that divides every thickness.",
            RuntimeWarning,
            stacklevel=2,
        )


def _grating_short_description(grating):
    if grating is None:
        return "no grating"
    stack = _grating_stack(grating)
    if stack is not None and hasattr(stack, "material_a") and hasattr(stack, "n_bilayers"):
        return (
            f"Grating {_material_label_text(stack.material_a)}/"
            f"{_material_label_text(stack.material_b)} ML"
        )
    layers = [_material_label_text(m) for m, _ in _grating_layers_bottom_up(grating)]
    if layers:
        return "Grating " + "/".join(reversed(layers))
    substrate = getattr(stack, "substrate_material", None)
    return f"Grating {_material_label_text(substrate)}" if substrate is not None else "Grating"


def _grating_text_lines(grating):
    """Compact description of a graxpy grating for the dashboard / printout."""
    if grating is None:
        return ["Grating            (none)"]
    lines = [
        f"Grating            {type(grating).__name__}",
        f"Grating period     {float(grating.period_lpermm):g} l/mm",
    ]
    blaze = getattr(grating, "blaze_angle_deg", None)
    if blaze is not None:
        lines.append(f"Blaze angle        {float(blaze):g}°")
        anti = getattr(grating, "anti_blaze_angle_deg", None)
        if anti is not None:
            lines.append(f"Anti-blaze angle   {float(anti):g}°")
    depth = getattr(grating, "depth_nm", None)
    if depth is not None:
        lines.append(f"Groove depth       {float(depth):.2f} nm")
    width = getattr(grating, "width_to_period_ratio", None)
    if width is not None:
        lines.append(f"Width/period       {float(width):g}")

    stack = _grating_stack(grating)
    if stack is not None and hasattr(stack, "material_a") and hasattr(stack, "n_bilayers"):
        lines += [
            f"Grating ML         {_material_label_text(stack.material_a)}/{_material_label_text(stack.material_b)}",
            f"Grating ML d       {float(stack.material_a_thickness_nm):g}/{float(stack.material_b_thickness_nm):g} nm",
            f"Grating ML periods {int(stack.n_bilayers):d}",
        ]
        if getattr(stack, "top_cap_material", None) is not None:
            lines.append(
                f"Top cap            {_material_label_text(stack.top_cap_material)} "
                f"{float(stack.top_cap_thickness_nm):g} nm"
            )
    else:
        for material, thickness in reversed(_grating_layers_bottom_up(grating)):
            lines.append(
                f"Layer              {_material_label_text(material)} / {float(thickness):g} nm"
            )
    substrate = getattr(stack, "substrate_material", None)
    if substrate is not None:
        lines.append(f"Grating substrate  {_material_label_text(substrate)}")
    lines.append(
        f"x/z resolution     {float(getattr(grating, 'x_resolution_nm', float('nan'))):g}/"
        f"{float(getattr(grating, 'z_resolution_nm', float('nan'))):g} nm"
    )
    roughness = getattr(grating, "roughness", None)
    if roughness is not None:
        lines.append(f"Roughness          {roughness}")
    lines.append("")
    return lines


def _resolve_worker_count(setting):
    """GraxPy's max_workers rules, also accepting any number as rcwa_workers."""
    if setting is None or isinstance(setting, str):
        return _resolve_max_workers(setting)
    return _resolve_max_workers(int(setting))


def _clean_efficiency(value):
    value = float(value)
    return float(np.clip(value, 0.0, 1.0)) if np.isfinite(value) else 0.0


def _grating_run_kwargs(p, fourier_orders):
    return dict(
        diffraction_order=int(p.diffraction_order),
        fourier_orders=int(fourier_orders),
        polarization="s",
        backend="numba",
        solver=str(p.rcwa_solver),
        validate_physical_results=False,
    )


def _grazing_angles_deg(energy_eV, cff, p):
    """Grating grazing angle (from the surface) in degrees; NaN if unphysical."""
    alpha, beta = solve_alpha_beta(energy_eV, cff, p)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    grazing = 90.0 - np.degrees(alpha)
    valid = np.isfinite(alpha) & np.isfinite(beta) & (grazing > 1e-6)
    return np.where(valid, grazing, np.nan)


def grating_efficiency_single(energy_eV, cff, p: SX700Parameters, harmonics=None):
    """
    Absolute efficiency of the selected diffraction order at (energy, c_ff).

    The grating is the graxpy object p.grating. Returns 0.0 where the grating
    equation has no solution. harmonics overrides p.rcwa_harmonics.
    """
    grating = _require_grating(p)
    _require_s_polarization(p)
    grazing = float(_grazing_angles_deg(float(energy_eV), float(cff), p))
    if not np.isfinite(grazing):
        return 0.0
    orders = p.rcwa_harmonics if harmonics is None else int(harmonics)
    result = grax.run_simulation(
        grating=grating,
        energy_ev=float(energy_eV),
        grazing_angle_deg=grazing,
        **_grating_run_kwargs(p, orders),
    )
    return _clean_efficiency(result.selected_efficiency)


def _iter_grating_points(p, points):
    """Yield (index, efficiency) for every (energy_eV, c_ff) in points.

    Unphysical points are yielded first with 0.0. The rest is solved serially
    (rcwa_workers == 1, in order) or with graxpy's BatchSimulationRunner
    (results arrive in completion order, hence the index).
    """
    grating = _require_grating(p)
    _require_s_polarization(p)
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    grazing = _grazing_angles_deg(pts[:, 0], pts[:, 1], p)
    valid = np.isfinite(grazing)
    for i in np.where(~valid)[0]:
        yield int(i), 0.0
    todo = np.where(valid)[0]
    if todo.size == 0:
        return

    run_kwargs = _grating_run_kwargs(p, p.rcwa_harmonics)
    workers = _resolve_worker_count(p.rcwa_workers)

    def context(i):
        return f"E={pts[i, 0]:.3f} eV, c_ff={pts[i, 1]:.4f}"

    if workers <= 1 or todo.size == 1:
        for i in todo:
            p.check_stopped()
            try:
                result = grax.run_simulation(
                    grating=grating,
                    energy_ev=float(pts[i, 0]),
                    grazing_angle_deg=float(grazing[i]),
                    **run_kwargs,
                )
            except Exception as exc:
                raise RuntimeError(f"graxpy failed at {context(i)}: {exc}") from exc
            yield int(i), _clean_efficiency(result.selected_efficiency)
        return

    runner = grax.BatchSimulationRunner(
        max_workers=workers,
        show_progress=False,
        on_error="continue",
        stop_event=p.stop_event,
        on_worker_pids_changed=p.worker_pids_callback,
        **run_kwargs,
    )
    cases = grax.energy_angle_cases(
        grating=grating,
        energy_angle_pairs=[(pts[i, 0], grazing[i]) for i in todo],
    )
    for result in runner.run_cases(cases):
        p.check_stopped()
        i = int(todo[int(result.index)])
        if result.status != "ok":
            raise RuntimeError(f"graxpy failed at {context(i)}: {result.error_message}")
        yield i, _clean_efficiency(result.selected_efficiency)
    p.check_stopped()


def _grating_fingerprint(grating):
    """Stable digest of a graxpy grating, incl. data a repr() would truncate."""
    digest = hashlib.sha1()
    digest.update(f"{type(grating).__module__}.{type(grating).__qualname__}".encode())
    digest.update(repr(grating).encode("utf-8"))
    try:
        import importlib.metadata as _metadata

        digest.update(_metadata.version("graxpy").encode("utf-8"))
    except Exception:
        pass

    def add_tables(value):
        # pandas DataFrames (user-defined n(E) tables) print truncated.
        if hasattr(value, "columns") and hasattr(value, "to_numpy"):
            digest.update(np.ascontiguousarray(value.to_numpy(dtype=float)).tobytes())

    for obj in (grating, _grating_stack(grating)):
        if obj is None:
            continue
        try:
            for value in vars(obj).values():
                add_tables(value)
        except TypeError:
            pass
    for material, _ in _grating_layers_bottom_up(grating):
        add_tables(material)
    return digest.hexdigest()
