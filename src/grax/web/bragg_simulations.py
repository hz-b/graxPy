"""Inputs, background run and saved results of Bragg-grating scans."""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from .bragg_gratings import coating_layers, default_solver, idt_grating_from_spec, surface_grating_from_spec
from .files import read_json, remove_file, write_json_atomically
from .jobs import _finish_active_run, request_stop, start_job, stop_event, stop_job
from .run_state import (
    _active_runs,
    _active_runs_lock,
    _elapsed_seconds,
    _eta_seconds,
    _update_active_run,
)

# One entry per kind of scan.
SCAN_TYPES: dict[str, dict[str, str]] = {
    "rocking_curve": {
        "label": "Rocking curve at different amplitudes",
        "summary": (
            "Rotates the grating at fixed photon energy, once for each deformation amplitude, and "
            "shows the reflected intensity with its satellites: through the crystal Bragg angle "
            "with Takagi-Taupin, or at grazing incidence, order by order, with RCWA or Nevière."
        ),
    },
    "energy_rocking_curve": {
        "label": "Rocking curve at different energies",
        "summary": (
            "Rotates the grating at one deformation amplitude, once for each photon energy: through "
            "the crystal Bragg angle of each energy with Takagi-Taupin, or at grazing incidence, "
            "order by order, with RCWA or Nevière, optionally following the multilayer peak."
        ),
    },
}
ENERGY_SCAN = "energy_rocking_curve"

SOLVERS: dict[str, str] = {
    "takagi": "Takagi-Taupin (crystal Bragg reflection)",
    "rcwa": "RCWA (grazing incidence)",
    "neviere": "Nevière differential method (grazing incidence)",
}
GRAZING_SOLVERS = ("rcwa", "neviere")

# Amplitudes that reproduce the rocking curves measured at 0 to 10 V in Fig. 4 of
# Vadilonga et al., Opt. Express 29, 34962 (2021).
PAPER_AMPLITUDES_NM = "0, 0.012, 0.024, 0.036, 0.048, 0.060, 0.072, 0.084, 0.096, 0.108, 0.120"

DEFAULTS: dict[str, Any] = {
    "scan_type": "rocking_curve",
    "energy_ev": 8000.0,
    "polarization": "sigma",
    "beam_x_um": 100.0,
    "beam_y_um": 100.0,
    "resolution_arcsec": 8.7,
    "amplitude_mode": "range",
    "amplitudes_nm": PAPER_AMPLITUDES_NM,
    "amplitude_start_nm": 0.0,
    "amplitude_stop_nm": 0.12,
    "amplitude_points": 11,
    "half_range_deg": 0.03,
    "step_deg": 0.0001,
    "n_orders": 17,
    "theta_center_deg": 0.5,
    "theta_half_range_deg": 0.45,
    "theta_step_deg": 0.002,
    "fourier_orders": 25,
    "workers": "auto",
    # Same sampling as the grating pages.
    "x_resolution_nm": 1.0,
    "z_resolution_nm": 1.0,
    "energy_mode": "range",
    "energies_ev": "8000, 10000, 12000",
    "energy_start_ev": 7000.0,
    "energy_stop_ev": 12000.0,
    "energy_points": 6,
    "amplitude_nm": 0.06,
}

_MAX_AMPLITUDES = 1000
_MAX_ENERGIES = 1000
# The range of the Henke optical tables.
MIN_ENERGY_EV = 10.0
MAX_ENERGY_EV = 30000.0
# Saved result fields that the run page plots.
_DATA_KEYS = ("angles_deg", "curves", "geometry", "orders", "total", "exit_angles_deg", "flat_reference")
_MAX_ANGLES = 100001
_partial_lock = threading.Lock()


def parse_params(form: Any, grating: dict[str, Any]) -> dict[str, Any]:
    """Read and validate the form for a saved grating; raise ``ValueError`` for the user."""

    def number(name: str, low: float, high: float, label: str) -> float:
        raw = str(form.get(name, DEFAULTS[name])).strip().replace(",", ".")
        try:
            value = float(raw)
        except ValueError:
            raise ValueError(f"{label} must be a number.") from None
        if not np.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{label} must be between {low:g} and {high:g}.")
        return value

    scan_type = str(form.get("scan_type", "rocking_curve"))
    if scan_type not in SCAN_TYPES:
        raise ValueError("Unknown scan type.")
    energy_scan = scan_type == ENERGY_SCAN
    if energy_scan:
        energies = _energies(form, number)
        amplitudes = [number("amplitude_nm", 0.0, 100.0, "Deformation amplitude")]
    mode = "single" if energy_scan else str(form.get("amplitude_mode", DEFAULTS["amplitude_mode"]))
    if mode == "single":
        pass
    elif mode == "range":
        start = number("amplitude_start_nm", 0.0, 100.0, "First amplitude")
        stop = number("amplitude_stop_nm", 0.0, 100.0, "Last amplitude")
        points = int(number("amplitude_points", 1, _MAX_AMPLITUDES, "Number of amplitudes"))
        amplitudes = [start] if points == 1 else [float(v) for v in np.linspace(start, stop, points)]
    elif mode == "list":
        try:
            amplitudes = [
                float(item) for item in str(form.get("amplitudes_nm", "")).replace(",", " ").split()
            ]
        except ValueError:
            raise ValueError("Amplitudes must be numbers separated by commas or spaces.") from None
    else:
        raise ValueError("Unknown way of giving the amplitudes.")
    if not amplitudes or len(amplitudes) > _MAX_AMPLITUDES:
        raise ValueError(f"Give between 1 and {_MAX_AMPLITUDES} amplitudes.")
    if any(not np.isfinite(value) or not 0.0 <= value <= 100.0 for value in amplitudes):
        raise ValueError("Amplitudes must lie between 0 and 100 nm.")
    polarization = str(form.get("polarization", "sigma")).strip().lower()
    if polarization not in {"sigma", "pi"}:
        raise ValueError("Polarization must be sigma or pi.")
    solver = str(form.get("solver", default_solver(grating))).strip().lower()
    if solver not in SOLVERS:
        raise ValueError(f"Choose a solver from {', '.join(SOLVERS)}.")
    if solver in GRAZING_SOLVERS:
        center = number("theta_center_deg", 1e-4, 90.0, "Central grazing angle")
        half_range = number("theta_half_range_deg", 1e-6, 90.0, "Grazing-angle half range")
        step = number("theta_step_deg", 1e-9, 90.0, "Grazing-angle step")
        start, stop = center - half_range, center + half_range
        if start <= 0.0 or stop > 90.0:
            raise ValueError("The grazing angles, centre ± half range, must lie between 0° and 90°.")
        if (stop - start) / step + 1.0 > _MAX_ANGLES:
            raise ValueError(f"The scan would have more than {_MAX_ANGLES} points; increase the step.")
        workers = str(form.get("workers", DEFAULTS["workers"])).strip().lower() or "auto"
        if workers != "auto":
            workers = str(int(number("workers", 1, 256, "Workers")))
        scan = {
            "theta_center_deg": center,
            "theta_half_range_deg": half_range,
            "theta_start_deg": start,
            "theta_stop_deg": stop,
            "theta_step_deg": step,
            "fourier_orders": int(number("fourier_orders", 1, 100, "Fourier orders")),
            "workers": workers,
            "x_resolution_nm": number("x_resolution_nm", 0.01, 1000.0, "X resolution"),
            "z_resolution_nm": number("z_resolution_nm", 0.01, 1000.0, "Z resolution"),
        }
        if energy_scan and str(form.get("theta_follow_peak", "")) == "on":
            from ..idt_surface_grating import multilayer_period_nm

            if multilayer_period_nm(coating_layers(grating)) is None:
                raise ValueError("This grating has no multilayer whose Bragg peak the scan could follow.")
            scan["theta_follow_peak"] = True
    else:
        n_orders = int(number("n_orders", 1, 101, "Lateral orders"))
        if n_orders % 2 == 0:
            raise ValueError("Lateral orders must be odd.")
        half_range = number("half_range_deg", 1e-4, 10.0, "Scan half range")
        step = number("step_deg", 1e-9, 10.0, "Scan step")
        if 2.0 * half_range / step + 1.0 > _MAX_ANGLES:
            raise ValueError(f"The scan would have more than {_MAX_ANGLES} points; increase the step.")
        scan = {"half_range_deg": half_range, "step_deg": step, "n_orders": n_orders}
    return {
        "grating_id": grating["id"],
        "grating": grating,
        "scan_type": scan_type,
        "energy_ev": energies[0] if energy_scan else number("energy_ev", MIN_ENERGY_EV, MAX_ENERGY_EV, "Photon energy"),
        **({"energies_ev": energies} if energy_scan else {}),
        "polarization": polarization,
        "beam_x_um": number("beam_x_um", 0.0, 100000.0, "Beam size in the scattering plane"),
        "beam_y_um": number("beam_y_um", 0.0, 100000.0, "Beam size across the scattering plane"),
        "resolution_arcsec": number("resolution_arcsec", 0.0, 3600.0, "Angular resolution"),
        "amplitudes_nm": sorted({round(value, 6) for value in amplitudes}),
        "solver": solver,
        **scan,
    }


def _energies(form: Any, number: Any) -> list[float]:
    """Photon energies of an energy scan, from a list or a linear range."""

    mode = str(form.get("energy_mode", DEFAULTS["energy_mode"]))
    if mode == "range":
        start = number("energy_start_ev", MIN_ENERGY_EV, MAX_ENERGY_EV, "First energy")
        stop = number("energy_stop_ev", MIN_ENERGY_EV, MAX_ENERGY_EV, "Last energy")
        points = int(number("energy_points", 1, _MAX_ENERGIES, "Number of energies"))
        energies = [start] if points == 1 else [float(v) for v in np.linspace(start, stop, points)]
    elif mode == "list":
        try:
            energies = [float(item) for item in str(form.get("energies_ev", "")).replace(",", " ").split()]
        except ValueError:
            raise ValueError("Energies must be numbers separated by commas or spaces.") from None
    else:
        raise ValueError("Unknown way of giving the energies.")
    if not energies or len(energies) > _MAX_ENERGIES:
        raise ValueError(f"Give between 1 and {_MAX_ENERGIES} energies.")
    if any(not np.isfinite(value) or not MIN_ENERGY_EV <= value <= MAX_ENERGY_EV for value in energies):
        raise ValueError(f"Energies must lie between {MIN_ENERGY_EV:g} and {MAX_ENERGY_EV:g} eV.")
    return sorted({round(value, 3) for value in energies})


def series(params: dict[str, Any]) -> dict[str, Any]:
    """What changes from curve to curve: the amplitudes, or the energies of an energy scan."""

    if params.get("scan_type") == ENERGY_SCAN:
        return {"values": params["energies_ev"], "label": "Photon energy (eV)", "unit": "eV", "name": "energies"}
    return {"values": params["amplitudes_nm"], "label": "Deformation amplitude (nm)", "unit": "nm", "name": "amplitudes"}


def angle_axis_label(params: dict[str, Any]) -> str:
    """Title of the angle axis of a scan's plots."""

    if params.get("solver", "takagi") not in GRAZING_SOLVERS:
        return "ΔΘ (°)"
    if params.get("theta_follow_peak"):
        return "Grazing angle from the multilayer peak (°)"
    return "Grazing angle (°)"


def default_theta_window(grating: dict[str, Any], energy_ev: float) -> dict[str, Any]:
    """Grazing-angle scan fields that frame the reflection of a coated grating.

    ``multilayer_peak_deg`` is the multilayer Bragg peak at this energy, or ``None``.
    """

    from ..idt_surface_grating import theta_window

    if not grating.get("coated"):
        return {}
    try:
        window = theta_window(surface_grating_from_spec(grating), energy_ev)
    except ValueError:
        return {}
    half = window["half_range_deg"]
    peak = window["multilayer_peak_deg"]
    return {
        "theta_center_deg": round(window["center_deg"], 4),
        "theta_half_range_deg": float(f"{half:.3g}"),
        "theta_step_deg": float(f"{2.0 * half / 400.0:.2g}"),
        "multilayer_peak_deg": None if peak is None else round(peak, 4),
        "critical_angle_deg": round(window["critical_deg"], 4),
    }


def simulations_dir(data_dir: Path) -> Path:
    """Folder that holds the saved scans."""

    return Path(data_dir) / "bragg_simulations"


def result_path(data_dir: Path, run_id: str) -> Path:
    """Saved result file of one scan."""

    return simulations_dir(data_dir) / f"{run_id}.json"


def compute_scan(params: dict[str, Any], progress: Any = None, stop: Any = None) -> dict[str, Any]:
    """Run the scan chosen in ``params``.

    ``progress(amplitude, angles_deg, curve)`` is called after each finished curve.
    """

    if params["scan_type"] == "rocking_curve":
        if params.get("solver", "takagi") in GRAZING_SOLVERS:
            return compute_surface_curves(params, progress, stop)
        return compute_rocking_curves(params, progress, stop)
    if params["scan_type"] == ENERGY_SCAN:
        if params.get("solver", "takagi") in GRAZING_SOLVERS:
            return compute_surface_energy_curves(params, progress, stop)
        return compute_energy_rocking_curves(params, progress, stop)
    raise ValueError(f"Unknown scan type {params['scan_type']!r}.")


def compute_rocking_curves(
    params: dict[str, Any], progress: Any = None, stop: Any = None
) -> dict[str, Any]:
    """Rocking curves for each amplitude, normalized to the peak of the undeformed crystal."""

    from ..bragg_grating import bragg_crystal, simulate_rocking_curves

    spec = params["grating"]
    crystal = bragg_crystal(
        spec["crystal"], (spec["h"], spec["k"], spec["l"]), params["energy_ev"]
    )
    half, step = params["half_range_deg"], params["step_deg"]
    angles_deg = np.arange(-half, half + step / 2.0, step)

    def on_curve(amplitude: float, curve: np.ndarray, peak_rad: float) -> None:
        if progress is not None:
            progress(amplitude, (angles_deg - np.degrees(peak_rad)).tolist(), curve.tolist())

    result = simulate_rocking_curves(
        crystal,
        idt_grating_from_spec(spec),
        params["amplitudes_nm"],
        np.radians(angles_deg),
        beam_x_um=params["beam_x_um"],
        beam_y_um=params["beam_y_um"],
        resolution_rad=np.radians(params["resolution_arcsec"] / 3600.0),
        n_orders=params["n_orders"],
        polarization=params["polarization"],
        coating=tuple((layer.material, layer.thickness_nm * 1e-3) for layer in coating_layers(spec)),
        progress=on_curve,
        stop=stop,
    )
    peak_deg = float(np.degrees(result["peak_angle_rad"]))
    return {
        "angles_deg": (angles_deg - peak_deg).tolist(),
        "curves": {f"{amplitude:g}": values.tolist() for amplitude, values in result["curves"].items()},
        "summary": {
            "theta_bragg_deg": float(np.degrees(crystal.theta_b_rad)),
            "d_spacing_angstrom": crystal.d_spacing_um * 1e4,
            "active_fraction": result["active_fraction"],
            "reference_peak_efficiency": result["reference_peak_efficiency"],
        },
    }


def compute_energy_rocking_curves(
    params: dict[str, Any], progress: Any = None, stop: Any = None
) -> dict[str, Any]:
    """Takagi-Taupin rocking curves at one amplitude for each energy.

    The angle is measured from the Bragg angle of each energy, and each curve is
    normalized to the peak of the undeformed crystal at that energy.
    """

    from ..bragg_grating import bragg_crystal, simulate_rocking_curves

    spec = params["grating"]
    amplitude = params["amplitudes_nm"][0]
    half, step = params["half_range_deg"], params["step_deg"]
    angles_deg = np.arange(-half, half + step / 2.0, step)
    idt = idt_grating_from_spec(spec)
    coating = tuple((layer.material, layer.thickness_nm * 1e-3) for layer in coating_layers(spec))
    curves: dict[str, list[float]] = {}
    bragg_angles: dict[str, float] = {}
    for energy in params["energies_ev"]:
        if stop is not None and stop.is_set():
            break
        crystal = bragg_crystal(spec["crystal"], (spec["h"], spec["k"], spec["l"]), energy)
        result = simulate_rocking_curves(
            crystal,
            idt,
            [amplitude],
            np.radians(angles_deg),
            beam_x_um=params["beam_x_um"],
            beam_y_um=params["beam_y_um"],
            resolution_rad=np.radians(params["resolution_arcsec"] / 3600.0),
            n_orders=params["n_orders"],
            polarization=params["polarization"],
            coating=coating,
        )
        curve = result["curves"][float(amplitude)]
        key = f"{energy:g}"
        curves[key] = curve.tolist()
        bragg_angles[key] = float(np.degrees(crystal.theta_b_rad))
        if progress is not None:
            progress(energy, angles_deg.tolist(), curves[key])
    return {
        "angles_deg": angles_deg.tolist(),
        "curves": curves,
        "summary": {"theta_bragg_deg": bragg_angles, "amplitude_nm": amplitude},
    }


def compute_surface_energy_curves(
    params: dict[str, Any], progress: Any = None, stop: Any = None
) -> dict[str, Any]:
    """Grazing-incidence rocking curves of every order at one amplitude, for each energy.

    With ``theta_follow_peak`` the window is centred on the multilayer Bragg peak of
    each energy and the angles are given from that peak; otherwise every energy
    scans the same absolute grazing angles.
    """

    from ..idt_surface_grating import surface_rocking_scan, theta_window

    spec = params["grating"]
    amplitude = params["amplitudes_nm"][0]
    half, step = params["theta_half_range_deg"], params["theta_step_deg"]
    offsets = np.arange(-half, half + step / 2.0, step)
    follow = bool(params.get("theta_follow_peak"))
    workers = params.get("workers", "auto")
    grating = surface_grating_from_spec(
        spec,
        x_resolution_nm=params.get("x_resolution_nm", DEFAULTS["x_resolution_nm"]),
        z_resolution_nm=params.get("z_resolution_nm", DEFAULTS["z_resolution_nm"]),
    )
    axis = offsets if follow else params["theta_center_deg"] + offsets
    curves: dict[str, dict[str, list]] = {}
    totals: dict[str, list] = {}
    exit_angles: dict[str, list] = {}
    centres: dict[str, float] = {}
    for energy in params["energies_ev"]:
        if stop is not None and stop.is_set():
            break
        key = f"{energy:g}"
        centre = theta_window(grating, energy)["multilayer_peak_deg"] if follow else params["theta_center_deg"]
        if centre - half <= 0.0:
            raise ValueError(
                f"At {energy:g} eV the window would reach below 0° (centre {centre:.4f}°); reduce the half range."
            )
        centres[key] = float(centre)

        def on_points(_amplitude, per_order, total, exits, computed, energy=energy):
            if progress is not None:
                progress(
                    energy,
                    axis.tolist(),
                    {
                        "orders": {str(order): _finite(values) for order, values in per_order.items()},
                        "total": _finite(total),
                        "exit_angles_deg": {str(order): _finite(values) for order, values in exits.items()},
                        "computed": computed,
                    },
                )

        result = surface_rocking_scan(
            grating,
            [amplitude],
            centre + offsets,
            energy_ev=energy,
            polarization="s" if params["polarization"] == "sigma" else "p",
            solver=params["solver"],
            fourier_orders=params["fourier_orders"],
            beam_x_um=params["beam_x_um"],
            beam_y_um=params["beam_y_um"],
            resolution_deg=params["resolution_arcsec"] / 3600.0,
            max_workers="auto" if workers == "auto" else int(workers),
            progress=on_points,
            stop=stop,
        )
        per_order = result["curves"].get(float(amplitude))
        if per_order is None:
            break
        curves[key] = {str(order): _finite(values) for order, values in per_order.items()}
        totals[key] = _finite(result["total"][float(amplitude)])
        exit_angles[key] = {str(order): _finite(values) for order, values in result["exit_angles_deg"].items()}
    orders = sorted({int(order) for per_order in curves.values() for order in per_order})
    return {
        "angles_deg": axis.tolist(),
        "geometry": "grazing",
        "orders": orders,
        "curves": curves,
        "total": totals,
        "exit_angles_deg": exit_angles,
        "summary": {"centre_deg": centres, "amplitude_nm": amplitude},
    }


def compute_surface_curves(
    params: dict[str, Any], progress: Any = None, stop: Any = None
) -> dict[str, Any]:
    """Grazing-incidence rocking curves of every diffraction order, for each amplitude.

    The intensities are absolute efficiencies (fractions of the incident beam).
    ``progress(amplitude, angles_deg, data)`` receives the curves of one amplitude.
    """

    from ..idt_surface_grating import surface_rocking_scan

    spec = params["grating"]
    start, stop_deg, step = params["theta_start_deg"], params["theta_stop_deg"], params["theta_step_deg"]
    angles_deg = np.arange(start, stop_deg + step / 2.0, step)
    workers = params.get("workers", "auto")

    def on_amplitude(amplitude: float, curves: dict, total: np.ndarray, exit_angles: dict, computed: int) -> None:
        if progress is not None:
            progress(
                amplitude,
                angles_deg.tolist(),
                {
                    "orders": {str(order): _finite(values) for order, values in curves.items()},
                    "total": _finite(total),
                    "exit_angles_deg": {str(order): _finite(values) for order, values in exit_angles.items()},
                    "computed": computed,
                },
            )

    result = surface_rocking_scan(
        surface_grating_from_spec(
            spec,
            x_resolution_nm=params.get("x_resolution_nm", DEFAULTS["x_resolution_nm"]),
            z_resolution_nm=params.get("z_resolution_nm", DEFAULTS["z_resolution_nm"]),
        ),
        params["amplitudes_nm"],
        angles_deg,
        energy_ev=params["energy_ev"],
        polarization="s" if params["polarization"] == "sigma" else "p",
        solver=params["solver"],
        fourier_orders=params["fourier_orders"],
        beam_x_um=params["beam_x_um"],
        beam_y_um=params["beam_y_um"],
        resolution_deg=params["resolution_arcsec"] / 3600.0,
        max_workers="auto" if workers == "auto" else int(workers),
        progress=on_amplitude,
        stop=stop,
    )
    return {
        "angles_deg": angles_deg.tolist(),
        "geometry": "grazing",
        "orders": result["orders"],
        "curves": {
            f"{amplitude:g}": {str(order): _finite(values) for order, values in per_order.items()}
            for amplitude, per_order in result["curves"].items()
        },
        "total": {f"{amplitude:g}": _finite(values) for amplitude, values in result["total"].items()},
        "exit_angles_deg": {str(order): _finite(values) for order, values in result["exit_angles_deg"].items()},
        "flat_reference": result["flat_reference"].tolist(),
        "summary": {"active_fraction": result["active_fraction"].tolist()},
    }


def _finite(values: np.ndarray) -> list[float | None]:
    """JSON-safe list: NaN (an order that does not propagate there) becomes ``None``."""

    return [None if not np.isfinite(value) else float(value) for value in values]


def _partials(app: Any) -> dict[str, dict[str, Any]]:
    return app.extensions.setdefault("grax_bragg_partial", {})


def start_scan(app: Any, data_dir: Path, params: dict[str, Any]) -> str:
    """Save the inputs, start the background calculation and return its run id."""

    run_id = uuid4().hex
    simulations_dir(data_dir).mkdir(parents=True, exist_ok=True)
    started = datetime.now().isoformat(timespec="seconds")
    write_json_atomically(
        result_path(data_dir, run_id), {"state": "running", "created_at": started, "params": params}
    )
    start_job(
        app,
        run_id,
        target=execute_scan,
        kwargs={
            "app": app,
            "data_dir": Path(data_dir),
            "run_id": run_id,
            "params": params,
            "created_at": started,
        },
        thread_name=f"grax-bragg-scan-{run_id}",
        workflow="bragg_scan",
        total_points=total_points(params),
    )
    return run_id


def angle_count(params: dict[str, Any]) -> int:
    """Number of grazing angles of a scan (one for a Takagi-Taupin scan, which reports whole curves)."""

    if params.get("solver", "takagi") not in GRAZING_SOLVERS:
        return 1
    start, stop, step = params["theta_start_deg"], params["theta_stop_deg"], params["theta_step_deg"]
    return int(np.arange(start, stop + step / 2.0, step).size)


def total_points(params: dict[str, Any]) -> int:
    """Progress points of a scan: amplitudes, times the angles for the grazing solvers."""

    return len(series(params)["values"]) * angle_count(params)


def execute_scan(
    app: Any, data_dir: Path, run_id: str, params: dict[str, Any], created_at: str = ""
) -> None:
    """Compute all amplitudes, publishing each curve as it finishes, then save the result."""

    from numba import get_num_threads

    stop = stop_event(app, run_id)
    computed: dict[str, int] = {}

    def publish(amplitude: float, angles_deg: list[float], curve: Any) -> None:
        key = f"{amplitude:g}"
        with _partial_lock:
            partial = _partials(app).setdefault(run_id, {"angles_deg": angles_deg, "curves": {}})
            if isinstance(curve, dict):
                partial["geometry"] = "grazing"
                partial["curves"][key] = curve["orders"]
                partial.setdefault("total", {})[key] = curve["total"]
                partial.setdefault("exit_angles_deg", {}).update(curve["exit_angles_deg"])
            else:
                partial["curves"][key] = curve
        # Grazing scans report point by point; a Takagi-Taupin curve is one point.
        done = curve.get("computed", 1) if isinstance(curve, dict) else 1
        increment = done - computed.get(key, 0)
        computed[key] = done
        if increment > 0:
            _update_active_run(app, run_id, completed_increment=increment)

    def save(payload: dict[str, Any], state: str) -> None:
        payload.update(
            {
                "state": state,
                "params": params,
                "created_at": created_at,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        write_json_atomically(result_path(data_dir, run_id), payload)

    try:
        _update_active_run(
            app, run_id, state="running", started=True, resolved_workers=get_num_threads()
        )
        payload = compute_scan(params, progress=publish, stop=stop)
        if stop.is_set():
            if payload["curves"]:
                save(payload, "aborted")
            _finish_active_run(app, run_id, state="aborted")
            return
        save(payload, "completed")
        _finish_active_run(app, run_id, state="completed")
    except Exception as error:
        app.logger.exception("Bragg scan %s failed", run_id)
        save({"error_text": str(error)}, "failed")
        _finish_active_run(app, run_id, state="failed", error_text=str(error))
    finally:
        with _partial_lock:
            _partials(app).pop(run_id, None)


def load_scan(data_dir: Path, run_id: str) -> dict[str, Any] | None:
    """The saved file of one scan (its inputs, and its curves once there are any)."""

    path = result_path(data_dir, run_id)
    if not path.exists():
        return None
    try:
        return read_json(path)
    except (OSError, ValueError):
        return None


def abort_scan(app: Any, run_id: str) -> bool:
    """Ask a running scan to stop after the curve it is computing."""

    return request_stop(app, run_id)


def list_scans(
    data_dir: Path, grating_id: str | None = None, limit: int | None = None
) -> list[dict[str, Any]]:
    """Saved scans, newest first, optionally only those of one grating."""

    folder = simulations_dir(data_dir)
    if not folder.exists():
        return []
    paths = sorted(folder.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    runs = []
    for path in paths:
        try:
            saved = read_json(path)
            params = saved["params"]
            if grating_id is not None and params["grating_id"] != grating_id:
                continue
            amplitudes = params["amplitudes_nm"]
            varied = series(params)
            values = varied["values"]
            runs.append(
                {
                    "id": path.stem,
                    "state": saved.get("state", "completed"),
                    "grating_id": params["grating_id"],
                    "grating": params["grating"]["name"],
                    "scan": SCAN_TYPES.get(params["scan_type"], {}).get("label", params["scan_type"]),
                    "solver": SOLVERS.get(params.get("solver", "takagi"), ""),
                    "when": saved.get("finished_at") or saved.get("created_at", ""),
                    "amplitudes": len(amplitudes),
                    "amplitude_range": f"{min(amplitudes):g} to {max(amplitudes):g} nm",
                    "series": (
                        f"{len(values)} {varied['name']}, {min(values):g} to {max(values):g} {varied['unit']}"
                        + (f" at {amplitudes[0]:g} nm" if varied["name"] == "energies" else "")
                    ),
                    "energy_ev": params["energy_ev"],
                    "beam": f"{params['beam_x_um']:g} × {params['beam_y_um']:g} µm",
                }
            )
        except (OSError, ValueError, KeyError):
            continue
        if limit is not None and len(runs) >= limit:
            break
    return runs


def recent_scans(data_dir: Path, grating_id: str, limit: int = 12) -> list[dict[str, Any]]:
    """The newest saved scans of one grating, for the list on its page."""

    return list_scans(data_dir, grating_id, limit)


def delete_scan(app: Any, data_dir: Path, run_id: str) -> None:
    """Stop a scan that is still running and remove its saved file."""

    stop_job(app, run_id)
    remove_file(result_path(data_dir, run_id))


def run_status(app: Any, data_dir: Path, run_id: str) -> dict[str, Any] | None:
    """Progress, timing and the curves so far of one scan; ``None`` if it is unknown.

    The field names of the progress part match what the shared run monitor reads.
    """

    saved = load_scan(data_dir, run_id)
    with _active_runs_lock(app):
        active = _active_runs(app).get(run_id)
        live = None
        if active is not None:
            live = {
                "state": active.state,
                "completed_points": active.completed_points,
                "total_points": active.total_points,
                "elapsed_seconds": _elapsed_seconds(active),
                "eta_seconds": _eta_seconds(active),
                "worker_mode": active.worker_mode,
                "requested_workers": active.requested_workers,
                "resolved_workers": active.resolved_workers,
                "error_text": active.error_text,
                "can_abort": active.state in {"queued", "running"} and not active.abort_requested,
            }
    if saved is None:
        return None
    data = None
    if saved.get("curves"):
        data = {key: saved[key] for key in _DATA_KEYS if key in saved}
    else:
        with _partial_lock:
            partial = _partials(app).get(run_id)
            if partial is not None:
                data = {key: (dict(value) if isinstance(value, dict) else value) for key, value in partial.items()}
    total = total_points(saved["params"])
    if live is None:
        state = saved.get("state", "completed")
        if state == "running":
            state = "aborted"
        live = {
            "state": state,
            "completed_points": _saved_points(data, saved["params"]),
            "total_points": total,
            "elapsed_seconds": None,
            "eta_seconds": 0.0,
            "worker_mode": "auto",
            "requested_workers": None,
            "resolved_workers": None,
            "error_text": saved.get("error_text", ""),
            "can_abort": False,
        }
    elif live["state"] in {"completed", "failed", "aborted"}:
        live["eta_seconds"] = 0.0
    live["remaining_points"] = max(live["total_points"] - live["completed_points"], 0)
    live["data"] = data
    return live


def _saved_points(data: dict[str, Any] | None, params: dict[str, Any]) -> int:
    """Points a saved scan holds: computed angles per amplitude for grazing scans, else curves."""

    if not data:
        return 0
    if data.get("geometry") != "grazing":
        return len(data.get("curves", {}))
    return sum(sum(value is not None for value in row) for row in data.get("total", {}).values())
