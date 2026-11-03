"""Adaptive (energy x c_ff) node sampling, interpolation and the grating-efficiency cache."""

import hashlib
from pathlib import Path

import numpy as np

from grax.materials import _load_henke_scattering_factors

from .grating import _grating_fingerprint, _grating_material_names, _require_grating
from .parameters import SX700Parameters


def _grating_henke_support_nodes(p, max_nodes):
    """Henke energies (incl. absorption edges) of the grating's element layers."""
    collected = []
    for name in _grating_material_names(_require_grating(p)):
        key = name.strip().capitalize()
        try:
            factors = _load_henke_scattering_factors(key)
        except Exception:
            continue
        collected.append(
            _select_henke_support_nodes(
                (factors.energy_ev, factors.f1, factors.f2),
                p.energy_min_eV,
                p.energy_max_eV,
                max_nodes,
            )
        )
    if not collected:
        return np.array([], dtype=float)
    nodes = np.unique(np.concatenate(collected))
    if nodes.size > max_nodes:
        nodes = nodes[np.linspace(0, nodes.size - 1, max_nodes).astype(int)]
    return nodes


def _select_henke_support_nodes(table, e_min, e_max, max_nodes):
    """Henke support points, preferring absorption edges."""
    e, f1, f2 = table
    mask = (e >= e_min) & (e <= e_max)
    idx = np.where(mask)[0]
    if len(idx) <= max_nodes:
        return e[idx]

    # Large jumps in f1/f2 mark edges; keep both sides.
    ee = e[idx]
    a1 = f1[idx]
    a2 = f2[idx]
    scale1 = np.maximum(1.0, np.maximum(np.abs(a1[:-1]), np.abs(a1[1:])))
    scale2 = np.maximum(1.0, np.maximum(np.abs(a2[:-1]), np.abs(a2[1:])))
    jump = np.maximum(
        np.abs(np.diff(a1)) / scale1,
        np.abs(np.diff(a2)) / scale2,
    )
    edge_local = np.where(jump > 0.03)[0]
    keep_local = set()
    for j in edge_local:
        keep_local.update([max(j - 1, 0), j, j + 1, min(j + 2, len(idx) - 1)])

    # Fill the rest with evenly spaced original points.
    remaining = max(0, max_nodes - len(keep_local))
    if remaining:
        uniform = np.linspace(0, len(idx) - 1, remaining, dtype=int)
        keep_local.update(map(int, uniform))

    keep_local = sorted(keep_local)
    if len(keep_local) > max_nodes:
        # Edges stay preferred; if there are very many, reduce evenly.
        pick = np.linspace(0, len(keep_local) - 1, max_nodes, dtype=int)
        keep_local = [keep_local[i] for i in pick]

    return ee[np.asarray(keep_local, dtype=int)]


def _interpolate_rect_grid(x_nodes, y_nodes, z_nodes, x_new, y_new):
    """Bilinear interpolation of a rectangular grid without SciPy."""
    x_nodes = np.asarray(x_nodes, float)
    y_nodes = np.asarray(y_nodes, float)
    z_nodes = np.asarray(z_nodes, float)
    x_new = np.asarray(x_new, float)
    y_new = np.asarray(y_new, float)

    tmp = np.vstack([np.interp(x_new, x_nodes, row) for row in z_nodes])
    out = np.empty((len(y_new), len(x_new)), dtype=float)
    for ix in range(len(x_new)):
        out[:, ix] = np.interp(y_new, y_nodes, tmp[:, ix])
    return out


def _rcwa_cache_file(p: SX700Parameters, e_nodes, c_nodes):
    cache_dir = (
        Path(p.cache_dir) if p.cache_dir is not None else Path.home() / ".cache" / "graxpy" / "pgm"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = (
        "sx700-v16-graxpy",
        _grating_fingerprint(_require_grating(p)),
        p.groove_density_l_mm,
        p.diffraction_order,
        p.polarization,
        p.rcwa_harmonics,
        p.rcwa_solver,
        tuple(np.round(e_nodes, 8)),
        tuple(np.round(c_nodes, 8)),
    )
    digest = hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"rcwa_{digest}.npz"


def _adaptive_refinement_scores(z, e_nodes, c_nodes):
    """Score E and c intervals by the largest normalized local map change."""
    z = np.asarray(z, float)
    scale = max(1e-12, float(np.nanpercentile(np.abs(np.where(np.isfinite(z), z, np.nan)), 90)))
    se = np.nanmax(np.abs(np.diff(z, axis=1)) / scale, axis=0) if z.shape[1] > 1 else np.array([])
    sc = np.nanmax(np.abs(np.diff(z, axis=0)) / scale, axis=1) if z.shape[0] > 1 else np.array([])
    return np.nan_to_num(se, nan=0.0), np.nan_to_num(sc, nan=0.0)


def _add_high_score_midpoints(nodes, scores, max_nodes, refine_fraction):
    nodes = np.asarray(nodes, float)
    if len(nodes) >= max_nodes or len(nodes) < 2:
        return nodes
    nadd = min(
        max_nodes - len(nodes),
        max(1, int(np.ceil((max_nodes - len(nodes)) * float(refine_fraction)))),
    )
    order = np.argsort(np.asarray(scores, float))[::-1]
    mids = []
    for i in order:
        if len(mids) >= nadd:
            break
        mids.append(0.5 * (nodes[i] + nodes[i + 1]))
    return np.unique(np.concatenate([nodes, np.asarray(mids)])) if mids else nodes


def _seed_adaptive_nodes(p, e_target, c_target, henke_nodes):
    ne = min(max(8, int(p.rcwa_adaptive_initial_energy_nodes)), e_target)
    nc = min(max(8, int(p.rcwa_adaptive_initial_cff_nodes)), c_target)
    e = np.linspace(p.energy_min_eV, p.energy_max_eV, ne)
    if len(henke_nodes):
        hn = np.asarray(henke_nodes, float)
        k = np.linspace(0, len(hn) - 1, min(len(hn), max(4, ne // 3)), dtype=int)
        e = np.unique(np.concatenate([e, hn[k]]))
    c = np.linspace(p.c_min, p.c_max, nc)
    if len(e) > e_target:
        e = np.linspace(e.min(), e.max(), e_target)
    if len(c) > c_target:
        c = np.linspace(c.min(), c.max(), c_target)
    return e, c
