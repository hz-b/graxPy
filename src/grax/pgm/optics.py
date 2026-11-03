# ruff: noqa: N806
# Preserve the colleague's scientific variable names and docstrings.
"""Mirror x grating efficiency maps."""

import time
import warnings

import numpy as np

from grax.planar_mirror import parratt_reflectivity

from .grating import (
    _iter_grating_points,
    _require_grating,
    _require_s_polarization,
    _resolve_worker_count,
    _warn_if_z_resolution_quantises,
    grating_efficiency_single,
)
from .parameters import SX700Parameters
from .sampling import (
    _adaptive_refinement_scores,
    _add_high_score_midpoints,
    _grating_henke_support_nodes,
    _interpolate_rect_grid,
    _rcwa_cache_file,
    _seed_adaptive_nodes,
)


def mirror_reflectivity_array(geom_data, p: SX700Parameters):
    """Evaluate GraxPy's saved mirror stack at the PGM geometry points."""
    if p.mirror_stack is None:
        raise ValueError("mirror_stack is required for optical efficiency calculations")
    energies = np.asarray(geom_data["E"], dtype=float)
    grazing = 90.0 - np.degrees(np.asarray(geom_data["theta"], dtype=float))
    valid = np.asarray(geom_data["optical"], dtype=bool) & np.isfinite(grazing)
    result = np.zeros(energies.shape, dtype=float)
    for column in range(energies.shape[1]):
        p.check_stopped()
        angle = np.clip(np.nan_to_num(grazing[:, column], nan=1e-6), 1e-6, 90.0)
        values = parratt_reflectivity(
            p.mirror_stack,
            float(energies[0, column]),
            angle,
            polarization="s",
            roughness_sigma_nm=p.mirror_roughness_nm,
        )
        result[:, column] = np.asarray(values, dtype=float).reshape(-1)
    return np.where(valid, result, 0.0)


def rcwa_convergence_report(p: SX700Parameters):
    """Automatic convergence check (number of Fourier orders) at representative points."""
    if not p.rcwa_convergence_check:
        return None

    # Generic in-range representative points; no reference curve is needed.
    e_samples = np.array(
        [
            p.energy_min_eV,
            np.sqrt(p.energy_min_eV * p.energy_max_eV),
            p.energy_max_eV,
        ],
        dtype=float,
    )
    c_samples = np.full_like(
        e_samples,
        0.5 * (p.c_min + p.c_max),
        dtype=float,
    )

    harmonic_diffs = []
    for E, cff in zip(e_samples, c_samples, strict=True):
        eta1 = grating_efficiency_single(E, cff, p, harmonics=p.rcwa_harmonics)
        eta2 = grating_efficiency_single(E, cff, p, harmonics=p.rcwa_harmonics + 2)
        harmonic_diffs.append(abs(eta2 - eta1))
    result = {}
    if harmonic_diffs:
        md = max(harmonic_diffs)
        result["harmonics_max_diff"] = md
        if p.rcwa_progress:
            print(
                f"RCWA convergence harmonics N={p.rcwa_harmonics} -> {p.rcwa_harmonics + 2}: "
                f"max difference {100 * md:.4f} percentage points "
                f"({len(harmonic_diffs)} representative points)"
            )
        if md > 0.01:
            warnings.warn(
                "RCWA harmonic convergence > 1 percentage point. Increase rcwa_harmonics.",
                RuntimeWarning,
                stacklevel=2,
            )
    return result if result else None


def calculate_optical_efficiency(p: SX700Parameters, energies, c_values, geom_data):
    """Calculate mirror R, grating eta, R*eta and R*eta*T_geom.

    Architecture:
      * the SX700 geometry and the mirror reflectivity (Parratt) are unchanged;
      * the grating efficiency comes from graxpy (RCWA or Neviere solver) for
        the grating object p.grating;
      * it is evaluated at the nodes of an adaptive (energy x c_ff) tensor grid,
        concentrating nodes where the efficiency map changes rapidly;
      * every node is solved only once: refinement stages reuse earlier nodes
        and the nodes are solved in parallel (rcwa_workers);
      * the final dashboard map is a smooth interpolation of the actual nodes.
    """
    grating = _require_grating(p)
    _require_s_polarization(p)
    _warn_if_z_resolution_quantises(grating)
    R_mirror = mirror_reflectivity_array(geom_data, p)
    if p.maps_ready_callback is not None:
        p.maps_ready_callback(energies, c_values, geom_data, R_mirror)

    e_target = max(10, int(p.rcwa_energy_nodes))
    c_target = max(10, int(p.rcwa_cff_nodes))

    if p.rcwa_include_henke_nodes:
        henke_nodes = _grating_henke_support_nodes(
            p, max(10, min(int(p.rcwa_max_henke_nodes), e_target))
        )
    else:
        henke_nodes = np.array([], dtype=float)

    if p.rcwa_adaptive_sampling:
        e_nodes, c_nodes = _seed_adaptive_nodes(p, e_target, c_target, henke_nodes)
    else:
        e_nodes = np.linspace(p.energy_min_eV, p.energy_max_eV, e_target)
        c_nodes = np.linspace(p.c_min, p.c_max, c_target)
        if henke_nodes.size:
            e_nodes = np.unique(np.concatenate([e_nodes, henke_nodes]))

    # Global timing state for the adaptive calculation. The initial grid gives
    # the first meaningful per-point timing. Later stages refine the estimate
    # from the actually measured throughput rather than assuming a fixed total
    # point count from the beginning.
    timing_state = {
        "wall_start": time.perf_counter(),
        "completed_points": 0,
        "completed_seconds": 0.0,
        "seed_points": None,
        "final_points_target": None,
        "stage_times": [],
        "estimated_total_seconds": None,
    }

    def _fmt_duration(seconds):
        sec = max(0.0, float(seconds))
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        ss = int(sec % 60)
        return f"{h}h {m:02d}m {ss:02d}s" if h else f"{m}m {ss:02d}s"

    def _update_global_estimate(known_remaining_upper_points=0):
        done_points = int(timing_state["completed_points"])
        elapsed = time.perf_counter() - timing_state["wall_start"]
        if done_points <= 0:
            return elapsed, None, None
        rate = elapsed / done_points
        current_upper = max(0, int(known_remaining_upper_points))
        estimated_total = elapsed + current_upper * rate
        timing_state["estimated_total_seconds"] = estimated_total
        return elapsed, max(0.0, estimated_total - elapsed), estimated_total

    node_cache = {}  # (energy_eV, c_ff) -> grating efficiency, shared by all stages

    def compute_nodes(
        e_nodes_local, c_nodes_local, stage_label="RCWA", global_remaining_upper_points=0
    ):
        cache_file = _rcwa_cache_file(p, e_nodes_local, c_nodes_local) if p.rcwa_use_cache else None
        if cache_file is not None and cache_file.exists():
            try:
                cached = np.load(cache_file)
                if np.array_equal(cached["e_nodes"], e_nodes_local) and np.array_equal(
                    cached["c_nodes"], c_nodes_local
                ):
                    if p.rcwa_progress:
                        print(f"RCWA cache used: {cache_file.name}")
                    timing_state["stage_times"].append(
                        (0.0, len(e_nodes_local) * len(c_nodes_local))
                    )
                    eta_cached = cached["eta_nodes"]
                    for ie, E in enumerate(e_nodes_local):
                        for ic, c in enumerate(c_nodes_local):
                            node_cache[(float(E), float(c))] = float(eta_cached[ic, ie])
                    return eta_cached
            except Exception:
                pass

        total_points = len(e_nodes_local) * len(c_nodes_local)
        wanted = [(float(E), float(c)) for E in e_nodes_local for c in c_nodes_local]
        missing = [key for key in dict.fromkeys(wanted) if key not in node_cache]
        reused = total_points - len(missing)
        done = 0
        next_percent = 10
        stage_t0 = time.perf_counter()

        if p.rcwa_progress:
            print(
                f"{stage_label}: {len(e_nodes_local)} energies x {len(c_nodes_local)} c_ff = "
                f"{total_points} points ({len(missing)} to solve, {reused} reused), "
                f"{2 * p.rcwa_harmonics + 1} orders, solver={p.rcwa_solver}, "
                f"workers={_resolve_worker_count(p.rcwa_workers)}"
            )

        for index, eta in _iter_grating_points(p, missing):
            p.check_stopped()
            node_cache[missing[index]] = eta
            done += 1
            timing_state["completed_points"] += 1
            if p.progress_callback is not None:
                p.progress_callback(int(timing_state["completed_points"]), e_target * c_target)
            if p.nodes_callback is not None:
                p.nodes_callback(
                    e_nodes_local,
                    c_nodes_local,
                    np.array(
                        [
                            [node_cache.get((float(E), float(c)), np.nan) for E in e_nodes_local]
                            for c in c_nodes_local
                        ],
                        dtype=float,
                    ),
                )
            if p.rcwa_progress:
                percent = int(100 * done / len(missing))
                while next_percent <= 100 and percent >= next_percent:
                    stage_elapsed = time.perf_counter() - stage_t0
                    stage_remain = (
                        stage_elapsed * (100.0 / next_percent - 1.0) if next_percent < 100 else 0.0
                    )
                    global_elapsed, global_remain, global_total = _update_global_estimate(
                        global_remaining_upper_points
                    )
                    if global_total is None:
                        eta_text = f"remaining {_fmt_duration(stage_remain)}"
                    else:
                        eta_text = (
                            f"remaining {_fmt_duration(stage_remain)}"
                            f" | global ETA ~{_fmt_duration(global_total)}"
                        )
                    print(
                        f"  RCWA {next_percent:3d}%  |  elapsed {_fmt_duration(stage_elapsed)}"
                        f"  |  {eta_text}"
                    )
                    next_percent += 10

        eta_nodes_local = np.zeros((len(c_nodes_local), len(e_nodes_local)), dtype=float)
        for ie, E in enumerate(e_nodes_local):
            for ic, c in enumerate(c_nodes_local):
                eta_nodes_local[ic, ie] = node_cache[(float(E), float(c))]

        stage_elapsed_total = time.perf_counter() - stage_t0
        timing_state["stage_times"].append((stage_elapsed_total, total_points))

        if cache_file is not None:
            np.savez_compressed(
                cache_file, e_nodes=e_nodes_local, c_nodes=c_nodes_local, eta_nodes=eta_nodes_local
            )
            if p.rcwa_progress:
                print(f"RCWA cache saved: {cache_file.name}")
        return eta_nodes_local

    seed_points = len(e_nodes) * len(c_nodes)
    timing_state["seed_points"] = seed_points
    timing_state["final_points_target"] = e_target * c_target

    eta_nodes = compute_nodes(
        e_nodes,
        c_nodes,
        "RCWA adaptive seed",
        global_remaining_upper_points=max(0, e_target * c_target - seed_points),
    )

    if p.rcwa_progress:
        elapsed = time.perf_counter() - timing_state["wall_start"]
        measured_rate = elapsed / max(1, timing_state["completed_points"])
        # Conservative upper-bound estimate: even if every remaining adaptive
        # step reaches the requested final tensor dimensions, this is the maximum
        # number of additional points that can still be required by the current
        # refinement strategy. The value is intentionally marked as approximate.
        iterations_left = max(0, int(p.rcwa_adaptive_iterations))
        conservative_extra = iterations_left * max(0, e_target * c_target - seed_points)
        est_total = elapsed + conservative_extra * measured_rate
        timing_state["estimated_total_seconds"] = est_total
        print(
            f"Initial RCWA grid complete | {seed_points} points | "
            f"elapsed {_fmt_duration(elapsed)} | "
            f"estimated total ~{_fmt_duration(est_total)} | "
            f"remaining ~{_fmt_duration(max(0.0, est_total - elapsed))}"
        )

    if p.rcwa_adaptive_sampling:
        iterations = max(0, int(p.rcwa_adaptive_iterations))
        for iteration in range(iterations):
            if len(e_nodes) >= e_target and len(c_nodes) >= c_target:
                break
            score_e, score_c = _adaptive_refinement_scores(eta_nodes, e_nodes, c_nodes)
            new_e = _add_high_score_midpoints(
                e_nodes, score_e, e_target, p.rcwa_adaptive_refine_fraction
            )
            new_c = _add_high_score_midpoints(
                c_nodes, score_c, c_target, p.rcwa_adaptive_refine_fraction
            )
            if len(new_e) == len(e_nodes) and len(new_c) == len(c_nodes):
                break
            e_nodes, c_nodes = new_e, new_c
            new_points = len(e_nodes) * len(c_nodes)
            # Only nodes that were not solved in an earlier stage cost time.
            # The global remaining estimate uses the maximum possible number of
            # future final-grid points.
            global_upper = max(0, e_target * c_target - new_points)
            if p.rcwa_progress:
                print(
                    f"Adaptive RCWA refinement {iteration + 1}/{iterations}: "
                    f"{len(e_nodes)} energies x {len(c_nodes)} c_ff = {new_points} points"
                )
            eta_nodes = compute_nodes(
                e_nodes,
                c_nodes,
                f"RCWA refine {iteration + 1}",
                global_remaining_upper_points=global_upper,
            )
            if p.rcwa_progress:
                elapsed = time.perf_counter() - timing_state["wall_start"]
                rate = elapsed / max(1, timing_state["completed_points"])
                total_estimate = elapsed + global_upper * rate
                print(
                    f"  Adaptive stage {iteration + 1} complete | "
                    f"elapsed {_fmt_duration(elapsed)} | "
                    f"remaining ~{_fmt_duration(max(0.0, total_estimate - elapsed))} | "
                    f"global ETA ~{_fmt_duration(total_estimate)}"
                )

    # Interpolate the exact grating nodes to the requested final plotting grid.
    eta_grating = _interpolate_rect_grid(e_nodes, c_nodes, eta_nodes, energies, c_values)
    eta_grating = np.where(geom_data["optical"], eta_grating, 0.0)
    optical_efficiency = np.clip(R_mirror * eta_grating, 0.0, 1.0)
    total_efficiency = np.clip(geom_data["throughput_fraction"] * optical_efficiency, 0.0, 1.0)

    result = {
        "mirror_reflectivity": R_mirror,
        "grating_efficiency": eta_grating,
        "optical_efficiency": optical_efficiency,
        "total_efficiency": total_efficiency,
        "optical_percent": 100.0 * optical_efficiency,
        "total_percent": 100.0 * total_efficiency,
        "rcwa_energy_nodes": e_nodes,
        "rcwa_cff_nodes": c_nodes,
        "rcwa_eta_nodes": eta_nodes,
        "rcwa_timing": {
            "elapsed_seconds": float(time.perf_counter() - timing_state["wall_start"]),
            "completed_points": int(timing_state["completed_points"]),
            "seed_points": int(timing_state["seed_points"] or 0),
            "target_points": int(timing_state["final_points_target"] or e_target * c_target),
            "estimated_total_seconds": (
                None
                if timing_state["estimated_total_seconds"] is None
                else float(timing_state["estimated_total_seconds"])
            ),
        },
    }
    if p.rcwa_convergence_check:
        p.check_stopped()
        result["rcwa_convergence"] = rcwa_convergence_report(p)
    return result
