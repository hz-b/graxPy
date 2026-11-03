# ruff: noqa: D103, D205, D212, D200, N803, N806
# Preserve the colleague's scientific variable names and docstrings.
"""SX700 geometry: grating equation, mirror/grating positions, footprints, working range."""

import numpy as np

from .constants import HC_EV_NM
from .parameters import SX700Parameters


def wavelength_mm(energy_eV):
    return (HC_EV_NM / energy_eV) * 1e-6


def solve_alpha_beta(energy_eV, cff, p: SX700Parameters):
    """
    Analytical PGM solution for alpha > 0, beta < 0.
    Returns radians.
    """
    energy_eV = np.asarray(energy_eV, dtype=float)
    cff = np.asarray(cff, dtype=float)

    lam = wavelength_mm(energy_eV)
    q = p.diffraction_order * lam * p.groove_density_l_mm

    den = 1.0 - cff**2
    with np.errstate(invalid="ignore", divide="ignore"):
        u = q / den
        sin_alpha = u + np.sqrt(1.0 + (u * cff) ** 2)

    physical = (energy_eV > 0) & (cff > 1.0) & np.isfinite(sin_alpha) & (np.abs(sin_alpha) <= 1.0)

    alpha = np.full(np.broadcast(energy_eV, cff).shape, np.nan, dtype=float)
    beta = np.full_like(alpha, np.nan)

    if np.any(physical):
        a = np.arcsin(np.clip(sin_alpha, -1.0, 1.0))
        cb = cff * np.cos(a)
        physical2 = physical & np.isfinite(cb) & (np.abs(cb) <= 1.0)

        alpha[physical2] = a[physical2]
        beta[physical2] = -np.arccos(np.clip(cb[physical2], -1.0, 1.0))

    if alpha.ndim == 0:
        return float(alpha), float(beta)
    return alpha, beta


def _cross2(ax, ay, bx, by):
    return ax * by - ay * bx


def _mirror_geometry_from_theta(theta, p: SX700Parameters):
    """
    2D SX700 geometry of the premirror.

    D = downstream end of the optical mirror surface.
    u = unit vector from D towards upstream along the mirror surface.

    The form matches the usual external SX700 rotation axis with
    the offsets a, c, h, v.
    """
    a = p.a_offset_mm
    c = p.c_offset_mm
    h = p.mirror_axis_dx_mm
    v = p.mirror_grating_axis_dy_mm

    st = np.sin(theta)
    ct = np.cos(theta)

    with np.errstate(divide="ignore", invalid="ignore"):
        cot_theta = ct / st

        Dx = -((a - c * cot_theta) * st) + h
        Dy = -(c / st + (a - c * cot_theta) * ct) + v

    ux = -st
    uy = -ct

    return Dx, Dy, ux, uy


def _point_to_segment_distance_2d(px, py, ax, ay, bx, by):
    """
    Minimal Euclidean distance from point P to finite line segment A-B.

    All coordinates may be NumPy arrays of identical/broadcastable shape.
    """
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay

    vv = vx * vx + vy * vy

    with np.errstate(divide="ignore", invalid="ignore"):
        t = (wx * vx + wy * vy) / vv

    t = np.clip(t, 0.0, 1.0)

    qx = ax + t * vx
    qy = ay + t * vy

    return np.hypot(px - qx, py - qy)


def _segment_to_segment_distance_2d(ax, ay, bx, by, cx, cy, dx, dy):
    """
    Minimal Euclidean distance between finite 2D segments A-B and C-D.

    The result is exactly zero if the two segments intersect.
    """
    d1 = _point_to_segment_distance_2d(ax, ay, cx, cy, dx, dy)
    d2 = _point_to_segment_distance_2d(bx, by, cx, cy, dx, dy)
    d3 = _point_to_segment_distance_2d(cx, cy, ax, ay, bx, by)
    d4 = _point_to_segment_distance_2d(dx, dy, ax, ay, bx, by)

    distance = np.minimum(
        np.minimum(d1, d2),
        np.minimum(d3, d4),
    )

    # Explicit finite-segment intersection test.
    rx = bx - ax
    ry = by - ay
    sx = dx - cx
    sy = dy - cy

    qpx = cx - ax
    qpy = cy - ay

    denominator = _cross2(rx, ry, sx, sy)
    eps = 1e-14

    with np.errstate(divide="ignore", invalid="ignore"):
        t = _cross2(qpx, qpy, sx, sy) / denominator
        u = _cross2(qpx, qpy, rx, ry) / denominator

    intersects = (np.abs(denominator) > eps) & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)

    distance = np.where(intersects, 0.0, distance)

    finite_geometry = (
        np.isfinite(ax)
        & np.isfinite(ay)
        & np.isfinite(bx)
        & np.isfinite(by)
        & np.isfinite(cx)
        & np.isfinite(cy)
        & np.isfinite(dx)
        & np.isfinite(dy)
    )

    return np.where(finite_geometry, distance, np.nan)


def calculate_touch_diff_deg(p: SX700Parameters):
    """
    Calculate the actual mirror/grating surface-angle difference at the
    configured critical-distance contour.

    The actual angle is
        touch_diff = |Theta - |beta||.

    Because d = crit_distance is normally a contour in (E, c_ff), touch_diff
    can vary slightly along that contour. A dense temporary geometry scan is
    used and the median contour value is displayed as the representative
    touch_diff parameter. The exact segment distance remains the mechanical
    criterion used by the plots.
    """
    nE = max(500, min(int(p.n_energy) * 2, 1600))
    nC = max(300, min(int(p.n_c) * 2, 1000))

    energies = np.linspace(p.energy_min_eV, p.energy_max_eV, nE)
    c_values = np.linspace(p.c_min, p.c_max, nC)
    data = geometry_arrays(energies, c_values, p)

    distance = np.asarray(data["mirror_grating_distance_mm"], dtype=float)
    delta = np.abs(
        np.asarray(data["theta_deg"], dtype=float) - np.asarray(data["beta_abs_deg"], dtype=float)
    )

    finite = np.asarray(data["optical"], dtype=bool) & np.isfinite(distance) & np.isfinite(delta)
    if not np.any(finite):
        return float("nan")

    target = float(p.crit_distance)
    abs_err = np.full_like(distance, np.inf, dtype=float)
    abs_err[finite] = np.abs(distance[finite] - target)
    min_err = float(np.min(abs_err))

    if not np.isfinite(min_err):
        return float("nan")

    # Select a narrow numerical band around d = crit_distance. The tolerance
    # is tied to the actual grid spacing, so the result is stable if the scan
    # resolution changes.
    d_finite = distance[finite]

    # A corresponding touch_diff exists only if the requested critical
    # distance is actually reached somewhere in the current scan.
    d_min = float(np.nanmin(d_finite))
    d_max = float(np.nanmax(d_finite))
    if target < d_min or target > d_max:
        return float("nan")

    unique_E = len(energies)
    unique_C = len(c_values)
    d_scale = max(1e-6, np.nanmax(d_finite) - np.nanmin(d_finite))
    tolerance = max(0.01, 2.0 * d_scale / max(unique_E, unique_C, 1))
    tolerance = max(tolerance, 3.0 * min_err)

    sel = finite & (np.abs(distance - target) <= tolerance)
    if not np.any(sel):
        idx = np.unravel_index(np.argmin(abs_err), abs_err.shape)
        return float(delta[idx])

    return float(np.nanmedian(delta[sel]))


def geometry_arrays(energies, c_values, p: SX700Parameters):
    """
    Fully vectorized calculation of the E-c_ff working range.
    """
    E, C = np.meshgrid(energies, c_values)

    alpha, beta = solve_alpha_beta(E, C, p)
    optical = np.isfinite(alpha) & np.isfinite(beta)

    theta = 0.5 * (alpha - beta)
    theta_g = np.pi / 2.0 - theta

    Dx, Dy, umx, umy = _mirror_geometry_from_theta(theta, p)

    # Three parallel rays: lower edge, centre, upper edge.
    y_center = -p.beam_offset_mm
    ys = np.array(
        [
            y_center - 0.5 * p.beam_height_mm,
            y_center,
            y_center + 0.5 * p.beam_height_mm,
        ]
    )

    mirror_s = []
    mirror_x = []
    grating_s = []
    grating_x = []
    grating_y = []
    forward_to_grating = []
    t_to_grating = []

    # Reflected direction; for a horizontal input:
    # delta = 180° - 2*Theta = 2*theta_g
    delta = np.pi - 2.0 * theta
    rdx = np.cos(delta)
    rdy = np.sin(delta)

    # Grating surface through O=(0,0)
    ugx = -np.sin(beta)
    ugy = np.cos(beta)

    # -------------------------------------------------------------
    # Minimum mechanical distance mirror <-> grating
    #
    # This calculates the exact minimum Euclidean distance between the
    # two FINITE 2D segments of the optical surfaces.
    # Holders, substrate thicknesses etc. are not included here.
    # -------------------------------------------------------------
    mirror_down_x = Dx
    mirror_down_y = Dy
    mirror_up_x = Dx + p.mirror_length_mm * umx
    mirror_up_y = Dy + p.mirror_length_mm * umy

    grating_half = 0.5 * p.grating_length_mm
    grating_1_x = -grating_half * ugx
    grating_1_y = -grating_half * ugy
    grating_2_x = +grating_half * ugx
    grating_2_y = +grating_half * ugy

    mirror_grating_distance_mm = _segment_to_segment_distance_2d(
        mirror_down_x,
        mirror_down_y,
        mirror_up_x,
        mirror_up_y,
        grating_1_x,
        grating_1_y,
        grating_2_x,
        grating_2_y,
    )

    distance_critical = (
        optical
        & np.isfinite(mirror_grating_distance_mm)
        & (mirror_grating_distance_mm < float(p.crit_distance))
    )
    distance_ok = optical & np.isfinite(mirror_grating_distance_mm) & ~distance_critical

    den_rg = _cross2(rdx, rdy, ugx, ugy)

    for y_ray in ys:
        with np.errstate(divide="ignore", invalid="ignore"):
            s_m = (y_ray - Dy) / umy
            x_m = Dx + s_m * umx

            # Intersection of the reflected ray with the grating line.
            # p + t*r = s_g*u_g
            cross_p_ug = _cross2(x_m, y_ray, ugx, ugy)
            t_rg = -cross_p_ug / den_rg

            x_g = x_m + t_rg * rdx
            y_g = y_ray + t_rg * rdy

            # Coordinate along the grating surface, relative to rotation axis O.
            s_g = x_g * ugx + y_g * ugy

        mirror_s.append(s_m)
        mirror_x.append(x_m)
        grating_s.append(s_g)
        grating_x.append(x_g)
        grating_y.append(y_g)
        forward_to_grating.append(t_rg > 0)
        t_to_grating.append(t_rg)

    mirror_s = np.stack(mirror_s, axis=0)
    mirror_x = np.stack(mirror_x, axis=0)
    grating_s = np.stack(grating_s, axis=0)
    grating_x = np.stack(grating_x, axis=0)
    grating_y = np.stack(grating_y, axis=0)
    forward_to_grating = np.stack(forward_to_grating, axis=0)
    t_to_grating = np.stack(t_to_grating, axis=0)

    m_min = p.edge_margin_mm
    m_max = p.mirror_length_mm - p.edge_margin_mm

    # -------------------------------------------------------------
    # Geometric transmission of the vertical beam height
    #
    # The mappings
    #   y_ray -> position on the mirror,
    #   y_ray -> position on the grating
    # are linear in this 2D geometry.
    #
    # The transmitted fraction of the beam height can therefore be calculated
    # exactly as the intersection of several allowed y intervals.
    #
    # throughput_fraction = 1.0  -> 100 % of the beam height passes
    # throughput_fraction = 0.0  -> 0 % passes
    #
    # This is a GEOMETRIC fraction of the vertical beam height.
    # For a non-uniform intensity profile (e.g. Gaussian) it is not
    # automatically identical to the transmitted photon flux.
    # -------------------------------------------------------------

    m_min = p.edge_margin_mm
    m_max = p.mirror_length_mm - p.edge_margin_mm
    g_half = 0.5 * p.grating_length_mm - p.edge_margin_mm

    # The earlier boolean checks stay for diagnostics and backward compatibility.
    mirror_ok = np.all(mirror_s >= m_min, axis=0) & np.all(mirror_s <= m_max, axis=0)
    mirror_center_ok = (mirror_s[1] >= m_min) & (mirror_s[1] <= m_max)

    grating_ok = np.all(np.abs(grating_s) <= g_half, axis=0)
    grating_center_ok = np.abs(grating_s[1]) <= g_half

    sequence_ok = np.all(forward_to_grating, axis=0)
    sequence_center_ok = forward_to_grating[1]

    theta_deg = np.degrees(theta)
    beta_deg = np.degrees(beta)
    beta_abs_deg = np.abs(beta_deg)

    theta_min_ok = theta_deg >= p.theta_min_deg
    theta_max_ok = theta_deg <= p.theta_max_deg
    theta_ok = theta_min_ok & theta_max_ok

    beta_min_ok = beta_abs_deg >= p.beta_min_deg
    beta_max_ok = beta_abs_deg <= p.beta_max_deg
    beta_ok = beta_min_ok & beta_max_ok

    y_beam_low = y_center - 0.5 * p.beam_height_mm
    y_beam_high = y_center + 0.5 * p.beam_height_mm

    # Start: the whole beam height is available in principle.
    y_allowed_low = np.full_like(E, y_beam_low, dtype=float)
    y_allowed_high = np.full_like(E, y_beam_high, dtype=float)

    # ---- Mirror interval ----------------------------------------
    # s_m = (y - Dy) / umy
    # The two mirror edges therefore correspond directly to two y limits.
    y_m1 = Dy + m_min * umy
    y_m2 = Dy + m_max * umy
    mirror_y_low = np.minimum(y_m1, y_m2)
    mirror_y_high = np.maximum(y_m1, y_m2)

    y_allowed_low = np.maximum(y_allowed_low, mirror_y_low)
    y_allowed_high = np.minimum(y_allowed_high, mirror_y_high)

    # ---- Grating interval ---------------------------------------
    # s_g(y) is linear. Slope from the lower and upper edge rays.
    if p.beam_height_mm > 0:
        slope_sg = (grating_s[2] - grating_s[0]) / p.beam_height_mm
    else:
        slope_sg = np.zeros_like(E)

    sg_center = grating_s[1]
    eps = 1e-14
    slope_nonzero = np.abs(slope_sg) > eps

    grating_y_low = np.full_like(E, -np.inf, dtype=float)
    grating_y_high = np.full_like(E, np.inf, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        yg1 = y_center + (-g_half - sg_center) / slope_sg
        yg2 = y_center + (g_half - sg_center) / slope_sg

    grating_y_low[slope_nonzero] = np.minimum(yg1, yg2)[slope_nonzero]
    grating_y_high[slope_nonzero] = np.maximum(yg1, yg2)[slope_nonzero]

    # If s_g is practically independent of y, either the whole beam
    # passes or none of it.
    sg_constant_outside = (~slope_nonzero) & (np.abs(sg_center) > g_half)
    grating_y_low[sg_constant_outside] = 1.0
    grating_y_high[sg_constant_outside] = 0.0

    y_allowed_low = np.maximum(y_allowed_low, grating_y_low)
    y_allowed_high = np.minimum(y_allowed_high, grating_y_high)

    # ---- Forward condition mirror -> grating ---------------------
    # t_rg(y) is also linear; t_rg > 0 is required.
    if p.beam_height_mm > 0:
        slope_t = (t_to_grating[2] - t_to_grating[0]) / p.beam_height_mm
    else:
        slope_t = np.zeros_like(E)

    t_center = t_to_grating[1]
    slope_t_nonzero = np.abs(slope_t) > eps

    seq_y_low = np.full_like(E, -np.inf, dtype=float)
    seq_y_high = np.full_like(E, np.inf, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        y_zero_t = y_center - t_center / slope_t

    # t(y) = t_center + slope_t * (y-y_center) > 0
    pos = slope_t > eps
    neg = slope_t < -eps
    seq_y_low[pos] = y_zero_t[pos]
    seq_y_high[neg] = y_zero_t[neg]

    t_constant_bad = (~slope_t_nonzero) & (t_center <= 0)
    seq_y_low[t_constant_bad] = 1.0
    seq_y_high[t_constant_bad] = 0.0

    y_allowed_low = np.maximum(y_allowed_low, seq_y_low)
    y_allowed_high = np.minimum(y_allowed_high, seq_y_high)

    # ---- Total geometric fraction --------------------------------
    if p.beam_height_mm > 0:
        transmitted_height = np.clip(
            y_allowed_high - y_allowed_low,
            0.0,
            p.beam_height_mm,
        )
        throughput_fraction = transmitted_height / p.beam_height_mm
    else:
        # Degenerate zero-height beam: the central ray decides.
        transmitted_height = np.zeros_like(E)
        throughput_fraction = (mirror_center_ok & grating_center_ok & sequence_center_ok).astype(
            float
        )

    # The angle and optics solutions apply to the complete beam state.
    global_state_ok = optical & theta_ok & beta_ok

    throughput_fraction = np.where(
        global_state_ok,
        throughput_fraction,
        0.0,
    )

    throughput_percent = 100.0 * throughput_fraction

    # For display in 5 % steps.
    throughput_percent_5 = 5.0 * np.round(throughput_percent / 5.0)
    throughput_percent_5 = np.clip(throughput_percent_5, 0.0, 100.0)

    # The earlier classification also stays available.
    center_ok = (
        optical & mirror_center_ok & grating_center_ok & sequence_center_ok & theta_ok & beta_ok
    )

    full_beam_ok = throughput_fraction >= (1.0 - 1e-12)
    partial_beam = (throughput_fraction > 0.0) & ~full_beam_ok
    center_outside = ~center_ok

    beam_class = np.zeros_like(E, dtype=np.uint8)
    beam_class[partial_beam] = 1
    beam_class[full_beam_ok] = 2

    # Backward compatibility:
    valid = full_beam_ok

    return {
        "E": E,
        "C": C,
        "alpha": alpha,
        "beta": beta,
        "theta": theta,
        "theta_g": theta_g,
        "alpha_deg": np.degrees(alpha),
        "beta_deg": beta_deg,
        "beta_abs_deg": beta_abs_deg,
        "theta_deg": theta_deg,
        "theta_g_deg": np.degrees(theta_g),
        "mirror_s": mirror_s,
        "mirror_x": mirror_x,
        "grating_s": grating_s,
        "grating_x": grating_x,
        "grating_y": grating_y,
        "mirror_grating_distance_mm": mirror_grating_distance_mm,
        "distance_critical": distance_critical,
        "distance_ok": distance_ok,
        "mirror_ok": mirror_ok,
        "mirror_center_ok": mirror_center_ok,
        "grating_ok": grating_ok,
        "grating_center_ok": grating_center_ok,
        "sequence_ok": sequence_ok,
        "sequence_center_ok": sequence_center_ok,
        "theta_min_ok": theta_min_ok,
        "theta_max_ok": theta_max_ok,
        "theta_ok": theta_ok,
        "beta_min_ok": beta_min_ok,
        "beta_max_ok": beta_max_ok,
        "beta_ok": beta_ok,
        "optical": optical,
        "center_ok": center_ok,
        "full_beam_ok": full_beam_ok,
        "partial_beam": partial_beam,
        "center_outside": center_outside,
        "beam_class": beam_class,
        "transmitted_height_mm": transmitted_height,
        "throughput_fraction": throughput_fraction,
        "throughput_percent": throughput_percent,
        "throughput_percent_5": throughput_percent_5,
        "valid": valid,
    }


def _cff_from_grating_grazing(energy_eV, grazing_deg, p: SX700Parameters):
    """Map grating-surface grazing angle -> SX700 c_ff at fixed energy."""
    alpha = np.pi / 2.0 - np.deg2rad(float(grazing_deg))
    lam_mm = wavelength_mm(float(energy_eV))
    q = float(p.diffraction_order) * lam_mm * float(p.groove_density_l_mm)
    sin_beta = q - np.sin(alpha)
    if not np.isfinite(sin_beta) or abs(sin_beta) > 1.0:
        return np.nan
    beta = np.arcsin(np.clip(sin_beta, -1.0, 1.0))
    if beta >= 0.0 or np.cos(alpha) <= 0.0:
        return np.nan
    cff = np.cos(beta) / np.cos(alpha)
    if not np.isfinite(cff) or cff <= 1.0:
        return np.nan
    return float(cff)


def geometry_for_state(energy_eV, cff, p: SX700Parameters):
    data = geometry_arrays(
        np.array([float(energy_eV)]),
        np.array([float(cff)]),
        p,
    )

    if not data["optical"][0, 0]:
        return None

    def scalar(name):
        return float(data[name][0, 0])

    def rayvec(name):
        return data[name][:, 0, 0].astype(float)

    g = {
        "valid": bool(data["valid"][0, 0]),
        "alpha_deg": scalar("alpha_deg"),
        "beta_deg": scalar("beta_deg"),
        "beta_abs_deg": scalar("beta_abs_deg"),
        "theta_deg": scalar("theta_deg"),
        "theta_grazing_deg": scalar("theta_g_deg"),
        "mirror_s_rays_mm": rayvec("mirror_s"),
        "mirror_x_rays_mm": rayvec("mirror_x"),
        "grating_s_rays_mm": rayvec("grating_s"),
        "grating_x_rays_mm": rayvec("grating_x"),
        "grating_y_rays_mm": rayvec("grating_y"),
        "mirror_ok": bool(data["mirror_ok"][0, 0]),
        "mirror_center_ok": bool(data["mirror_center_ok"][0, 0]),
        "grating_ok": bool(data["grating_ok"][0, 0]),
        "grating_center_ok": bool(data["grating_center_ok"][0, 0]),
        "sequence_ok": bool(data["sequence_ok"][0, 0]),
        "sequence_center_ok": bool(data["sequence_center_ok"][0, 0]),
        "theta_min_ok": bool(data["theta_min_ok"][0, 0]),
        "theta_max_ok": bool(data["theta_max_ok"][0, 0]),
        "theta_ok": bool(data["theta_ok"][0, 0]),
        "beta_min_ok": bool(data["beta_min_ok"][0, 0]),
        "beta_max_ok": bool(data["beta_max_ok"][0, 0]),
        "beta_ok": bool(data["beta_ok"][0, 0]),
        "center_ok": bool(data["center_ok"][0, 0]),
        "full_beam_ok": bool(data["full_beam_ok"][0, 0]),
        "partial_beam": bool(data["partial_beam"][0, 0]),
        "beam_class": int(data["beam_class"][0, 0]),
        "transmitted_height_mm": float(data["transmitted_height_mm"][0, 0]),
        "throughput_fraction": float(data["throughput_fraction"][0, 0]),
        "throughput_percent": float(data["throughput_percent"][0, 0]),
        "throughput_percent_5": float(data["throughput_percent_5"][0, 0]),
    }

    g["beam_status"] = (
        "vollständig innerhalb"
        if g["throughput_percent"] >= 100.0 - 1e-9
        else "vollständig abgeschnitten"
        if g["throughput_percent"] <= 1e-9
        else f"{g['throughput_percent']:.1f} % geometrischer Durchlass"
    )

    g["mirror_hit_x_mm"] = g["mirror_x_rays_mm"][1]
    g["mirror_hit_y_mm"] = -p.beam_offset_mm
    g["mirror_hit_from_downstream_edge_mm"] = g["mirror_s_rays_mm"][1]

    g["mirror_footprint_mm"] = np.max(g["mirror_s_rays_mm"]) - np.min(g["mirror_s_rays_mm"])
    g["grating_footprint_mm"] = np.max(g["grating_s_rays_mm"]) - np.min(g["grating_s_rays_mm"])
    g["grating_center_error_mm"] = abs(g["grating_s_rays_mm"][1])

    return g


def calculate_working_range(p: SX700Parameters):
    energies = np.linspace(p.energy_min_eV, p.energy_max_eV, p.n_energy)
    c_values = np.linspace(p.c_min, p.c_max, p.n_c)
    data = geometry_arrays(energies, c_values, p)
    return energies, c_values, data
