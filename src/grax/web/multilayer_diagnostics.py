"""Interactive diagnostics (theta scans, survey, energy plots) for multilayer design studies."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
from grax.multilayer_design import MultilayerGratingDesigner, _label_indices

from .multilayer_design_studies import build_design_config


def _descriptor(plot_id: str, label: str, kind: str, source: Path) -> dict[str, Any]:
    return {"id": plot_id, "label": label, "kind": kind, "source": source}


def discover_plots(root: Path, current: Path, config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return logical plots available at ``current`` inside ``root``."""
    config = config or {}
    plots: list[dict[str, Any]] = []
    survey_csv = root / "survey" / "survey.csv"
    if current == root and survey_csv.is_file():
        plots += [
            _descriptor("survey-optimal", f"Optimal {_profile_name(config)} versus d-spacing", "survey_optimal", survey_csv),
            _descriptor("survey-maximum", "Maximum efficiency versus d-spacing", "survey_maximum", survey_csv),
            _descriptor("survey-heatmap", "Efficiency heatmap", "survey_heatmap", survey_csv),
        ]
    summaries = sorted((root / "energy_scan").glob("*/multilayer_theta_search_summary.csv"))
    if current == root:
        if len(summaries) > 1:
            plots.append(_descriptor("energy-comparison", "Energy-scan comparison", "energy_comparison", root / "energy_scan"))
        for summary in summaries:
            plots.append(_descriptor(f"energy:{summary.parent.name}", f"Efficiency versus energy — {summary.parent.name}", "energy", summary))
    summary = current / "multilayer_theta_search_summary.csv"
    all_orders = current / "multilayer_theta_search_all_orders.csv"
    if summary.is_file():
        plots.append(_descriptor("energy", "Efficiency versus energy", "energy", summary))
        theta_files = sorted((current / "theta_scans").glob("theta_scan_*.csv"))
        if theta_files:
            plots.append(_descriptor("workflow", "Theta-search workflow", "workflow", summary))
    if all_orders.is_file():
        plots.append(_descriptor("all-orders", "All diffraction orders", "all_orders", all_orders))
    for filename, plot_id, label, kind in (
        ("blazed_multilayer_profile.png", "profile", "Grating profile and material map", "profile"),
        ("multilayer_stack_schematic.png", "stack", "Multilayer stack schematic", "stack"),
    ):
        if (current / filename).is_file() and not any(item["id"] == plot_id for item in plots):
            plots.append(_descriptor(plot_id, label, kind, current))
    theta_files = sorted(current.rglob("theta_scan_*.csv"))
    if theta_files:
        plots.insert(0, _descriptor("theta-all", "Plot all theta searches in this folder", "theta_all", current))
    return plots


def _finite(values) -> list[float]:
    return [float(value) for value in values if pd.notna(value) and math.isfinite(float(value))]


def _base_layout(title: str, x: str, y: str) -> dict[str, Any]:
    return {"template": "plotly_white", "title": {"text": title, "x": 0.02},
            "xaxis": {"title": x}, "yaxis": {"title": y}, "hovermode": "closest",
            "showlegend": True, "legend": {"orientation": "h", "y": -0.18},
            "margin": {"l": 70, "r": 40, "t": 85, "b": 90}}


def _profile_name(config: dict[str, Any]) -> str:
    return "grating depth" if config.get("grating_type", "blazed") == "laminar" else "blaze angle"


def _profile_label(config: dict[str, Any]) -> str:
    unit = "nm" if config.get("grating_type", "blazed") == "laminar" else "deg"
    return f"{_profile_name(config).capitalize()} ({unit})"


def _number(value: Any, default: float) -> float:
    """Coerce old or incomplete mapping metadata without breaking diagnostics."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _survey_plot(kind: str, csv_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    table = pd.read_csv(csv_path).dropna(subset=["d_nm", "blaze_deg", "peak_efficiency"])
    best = table.loc[table.groupby("d_nm")["peak_efficiency"].idxmax()].sort_values("d_nm")
    profile_label = _profile_label(config)
    profile_name = _profile_name(config)
    energy = _number(config.get("target_energy_ev"), 9000.0)
    order = int(_number(config.get("diffraction_order"), 2))
    if kind == "survey_optimal":
        labels = [f"{value:.3g}" if index in _label_indices(best.peak_efficiency.to_numpy()) else ""
                  for index, value in enumerate(best.peak_efficiency)]
        trace = {"type": "scatter", "mode": "lines+markers+text", "x": _finite(best.d_nm),
                 "y": _finite(best.blaze_deg), "text": labels,
                 "textposition": "top right", "line": {"color": "#999"},
                 "marker": {"size": 10, "color": _finite(best.peak_efficiency), "colorscale": "Viridis",
                            "colorbar": {"title": "peak efficiency"}}, "name": "optimal"}
        return {"traces": [trace], "layout": _base_layout(
            f"Optimal {profile_name} versus d-spacing at {energy:g} eV "
            f"(order {order}; colour/labels: peak efficiency)",
            "Bilayer d-spacing (nm)", f"Optimal {profile_label}")}
    if kind == "survey_maximum":
        label_indices = _label_indices(best.peak_efficiency.to_numpy(), change_series=best.blaze_deg.to_numpy())
        unit = "nm" if profile_name == "grating depth" else "deg"
        labels = [f"{profile_name} {value:.3f} {unit}" if index in label_indices else ""
                  for index, value in enumerate(best.blaze_deg)]
        trace = {"type": "scatter", "mode": "lines+markers+text", "x": _finite(best.d_nm),
                 "y": _finite(best.peak_efficiency), "text": labels,
                 "textposition": "top right", "line": {"color": "#999"},
                 "marker": {"size": 10, "color": _finite(best.blaze_deg), "colorscale": "Plasma",
                            "colorbar": {"title": f"optimal {profile_label.lower()}"}}, "name": "maximum"}
        return {"traces": [trace], "layout": _base_layout(
            f"Max efficiency versus d-spacing at {energy:g} eV "
            f"(order {order}; colour/labels: optimal {profile_name})",
            "Bilayer d-spacing (nm)", f"Max selected-order efficiency ({config.get('polarization', 'p')}-pol)")}
    d_values = sorted(float(v) for v in table.d_nm.unique())
    p_values = sorted(float(v) for v in table.blaze_deg.unique())
    lookup = {(float(r.d_nm), float(r.blaze_deg)): float(r.peak_efficiency) for r in table.itertuples()}
    z = [[lookup.get((d, profile)) for d in d_values] for profile in p_values]
    heatmap = {"type": "heatmap", "x": d_values, "y": p_values, "z": z, "colorscale": "Viridis",
               "colorbar": {"title": f"peak selected-order efficiency ({config.get('polarization', 'p')}-pol)"},
               "hovertemplate": "d=%{x:.3f} nm<br>profile=%{y:.3f}<br>efficiency=%{z:.4f}<extra></extra>"}
    ridge = {"type": "scatter", "mode": "lines+markers", "x": _finite(best.d_nm), "y": _finite(best.blaze_deg),
             "line": {"color": "white", "width": 2}, "marker": {"color": "white", "line": {"color": "black", "width": 1}}, "name": "optimal per d"}
    compact_profile = "depth" if profile_name == "grating depth" else "blaze"
    return {"traces": [heatmap, ridge], "layout": _base_layout(f"Peak efficiency over (d, {compact_profile}) at {energy:g} eV (order {order})", "Bilayer d-spacing (nm)", profile_label)}


def _energy_plot(paths: list[Path], config: dict[str, Any], comparison: bool) -> dict[str, Any]:
    traces = []
    profile_name = _profile_name(config)
    profile_unit = "nm" if profile_name == "grating depth" else "deg"
    for path in paths:
        table = pd.read_csv(path).dropna(subset=["energy_ev", "selected_efficiency"])
        d_spacing, profile = _design_values(path.parent)
        traces.append({"type": "scatter", "mode": "lines", "line": {"width": 1.6},
                       "x": _finite(table.energy_ev), "y": _finite(table.selected_efficiency),
                       "name": f"d = {d_spacing:.3f} nm, {profile_name} = {profile:.3f} {profile_unit}"})
    material_a = config.get("material_a", ["A"])[0]
    material_b = config.get("material_b", ["B"])[0]
    coating = config.get("coating_label") or f"{material_a}/{material_b}"
    order = int(_number(config.get("diffraction_order"), 2))
    if comparison:
        title = f"{coating} multilayer grating (order {order}): {len(paths)} designs"
    else:
        d_spacing, profile = _design_values(paths[0].parent)
        title = (f"{coating} multilayer grating (order {order}): d = {d_spacing:.3f} nm, "
                 f"{profile_name} = {profile:.3f} {profile_unit}")
    return {"traces": traces, "layout": _base_layout(title, "Photon energy (eV)", f"Selected-order efficiency ({config.get('polarization', 'p')}-pol)")}


def _all_orders(path: Path) -> dict[str, Any]:
    table = pd.read_csv(path).dropna(subset=["energy_ev", "order", "efficiency"])
    traces = [{"type": "scatter", "mode": "lines", "x": _finite(part.energy_ev), "y": _finite(part.efficiency), "name": f"Order {int(order)}"} for order, part in table.groupby("order")]
    return {"traces": traces, "layout": _base_layout("All diffraction orders versus energy", "Photon energy (eV)", "Diffraction efficiency")}


_PALE_SCAN_COLORS = {"rough": "#a6cee3", "precise": "#fdbf6f"}
_THETA_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#d62728")


def _natural_key(path: Path) -> list[Any]:
    """Sort ``d3.5nm`` before ``d10nm`` and ``3000eV`` before ``12000eV``."""
    return [float(part) if index % 2 else part for index, part in enumerate(re.split(r"(\d+\.\d+|\d+)", path.as_posix()))]


def _design_name(run_dir: Path) -> str:
    """Name a design folder with its d-spacing (survey runs keep it one folder up)."""
    if re.match(r"d[0-9.]+nm$", run_dir.parent.name):
        return f"{run_dir.parent.name}_{run_dir.name}"
    return run_dir.name


def _theta_traces(directory: Path, *, legend: bool = True, paths: list[Path] | None = None) -> list[dict[str, Any]]:
    """Return one group of traces per theta-scan file.

    Each trace carries ``group`` (index of its scan file) and ``role`` (``scan``, ``peak`` or
    ``legend``); the page uses them to show or hide a scan, its selected peak and its single
    legend entry together. ``scan_class`` (rough/precise/fit) drives the scan-type checkboxes.
    """
    traces = []
    for group, path in enumerate(paths or sorted(directory.rglob("theta_scan_*.csv"), key=_natural_key)):
        table = pd.read_csv(path).dropna(subset=["grazing_angle_deg", "selected_efficiency"])
        label = f"{_design_name(path.parent.parent)} · {path.stem}"
        color = _THETA_COLORS[group % len(_THETA_COLORS)]
        if legend:
            traces.append({"type": "scatter", "mode": "lines+markers", "x": [None], "y": [None], "name": label,
                           "line": {"width": 1, "color": color}, "marker": {"size": 5, "color": color},
                           "group": group, "role": "legend", "scan_class": "legend"})
        for scan_type, part in table.groupby("scan_type", sort=False):
            if str(scan_type).lower() == "peak":
                continue  # the fitted peak: drawn below as the selected-peak star
            scan_class = {"rough": "rough", "fit": "fit"}.get(str(scan_type).lower(), "precise")
            if scan_class == "fit":
                traces.append({"type": "scatter", "mode": "lines", "x": _finite(part.grazing_angle_deg),
                               "y": _finite(part.selected_efficiency), "name": f"{label} · fit",
                               "line": {"width": 2.5, "color": "red"},
                               "showlegend": not legend, "group": group, "role": "scan", "scan_class": "fit"})
                continue
            # Pale, so the fit and the peak markers stand out: a scan per colour when many are
            # drawn together, a fixed pale colour per pass when one cell is shown alone.
            scan_color = color if legend else _PALE_SCAN_COLORS[scan_class]
            traces.append({"type": "scatter", "mode": "lines+markers", "x": _finite(part.grazing_angle_deg),
                           "y": _finite(part.selected_efficiency), "name": f"{label} · {scan_type}",
                           "opacity": (0.4 if scan_class == "rough" else 0.65) if legend else 1,
                           "line": {"width": 1, "color": scan_color},
                           "marker": {"size": 5, "color": scan_color, "symbol": "circle" if scan_class == "rough" else "square"},
                           "showlegend": not legend, "group": group, "role": "scan", "scan_class": scan_class})
        precise = table[table["scan_type"].str.lower() == "precise"]
        if not precise.empty:
            top = precise.loc[precise["selected_efficiency"].idxmax()]
            selected_rows = table[table["is_selected_peak"].astype(bool)] if "is_selected_peak" in table.columns else table.iloc[0:0]
            if not any(math.isclose(float(top.grazing_angle_deg), float(x), abs_tol=1e-9) for x in selected_rows.grazing_angle_deg):
                traces.append({"type": "scatter", "mode": "markers", "x": [float(top.grazing_angle_deg)],
                               "y": [float(top.selected_efficiency)], "name": f"{label} · maximum",
                               "marker": {"size": 11, "symbol": "diamond-open", "color": "black", "line": {"width": 2}},
                               "showlegend": not legend, "group": group, "role": "peak", "scan_class": "precise"})
        if "is_selected_peak" in table.columns:
            selected = table[table["is_selected_peak"].astype(bool)]
            if not selected.empty:
                traces.append({"type": "scatter", "mode": "markers", "x": _finite(selected.grazing_angle_deg),
                               "y": _finite(selected.selected_efficiency), "name": f"{label} · selected peak",
                               "marker": {"size": 12, "symbol": "star", "color": "red"},
                               "showlegend": not legend, "group": group, "role": "peak", "scan_class": "precise"})
    # Plotly paints in trace order: scans first, then the fit, then the peak markers on top.
    def paint_order(trace: dict[str, Any]) -> int:
        if trace["role"] == "legend":
            return 0
        if trace["role"] == "peak":
            return 4 if trace["name"].endswith("selected peak") else 3
        return {"rough": 1, "precise": 1, "fit": 2}[trace["scan_class"]]

    return sorted(traces, key=paint_order)


def survey_cell_scans(
    study_dir: Path, d_nm: float | None = None, blaze_deg: float | None = None, *, latest: bool = False
) -> dict[str, Any]:
    """Return the rough scan, fine scan, fit and selected peak of one survey cell.

    A cell is named by ``d_nm`` and ``blaze_deg``. Without them the newest solved cell is shown
    when ``latest`` is set (a survey still running) and the best cell otherwise. An empty
    ``traces`` list means no cell has stored a scan yet.
    """
    runs = Path(study_dir) / "survey" / "runs"
    cell: Path | None = None
    if d_nm is not None and blaze_deg is not None:
        matches = list((runs / f"d{d_nm:.3f}nm").glob(f"*{blaze_deg:.3f}*"))
        cell = matches[0] if matches else None
    elif latest:
        scans = list(runs.glob("*/*/theta_scans/theta_scan_*.csv"))
        cell = max(scans, key=lambda path: path.stat().st_mtime).parent.parent if scans else None
    else:
        survey_csv = Path(study_dir) / "survey" / "survey.csv"
        try:
            table = pd.read_csv(survey_csv).dropna(subset=["d_nm", "blaze_deg", "peak_efficiency"])
            best = table.loc[table["peak_efficiency"].idxmax()]
            return survey_cell_scans(study_dir, float(best.d_nm), float(best.blaze_deg))
        except (OSError, ValueError, KeyError):
            return {"traces": [], "layout": {}}
    if cell is None or not cell.is_dir():
        return {"traces": [], "layout": {}}
    traces = _theta_traces(cell, legend=False)
    for trace in traces:
        for key in ("group", "role", "scan_class"):
            trace.pop(key, None)
        trace["showlegend"] = True
    if not traces:
        return {"traces": [], "layout": {}}
    d_value, profile = _design_values(cell.relative_to(Path(study_dir)))
    layout = _base_layout(f"Theta scans: d = {d_value:.3f} nm, profile = {profile:.3f}",
                          "Grazing angle (deg)", "Diffraction efficiency")
    return {"traces": traces, "layout": layout, "d_nm": d_value, "blaze_deg": profile}


def energy_scan_scans(
    study_dir: Path, d_nm: float | None = None, blaze_deg: float | None = None,
    energy_ev: float | None = None, *, latest: bool = False,
) -> dict[str, Any]:
    """Return the rough scan, fine scan, fit and selected peak of one energy-scan point.

    The point is named by its design (``d_nm``, ``blaze_deg``) and ``energy_ev``. Without them the
    newest stored scan is shown when ``latest`` is set (a scan still running), and otherwise the
    first design at its most efficient energy. An empty ``traces`` list means nothing is stored yet.
    """
    root = Path(study_dir) / "energy_scan"
    empty: dict[str, Any] = {"traces": [], "layout": {}}
    path: Path | None = None
    if d_nm is not None and blaze_deg is not None and energy_ev is not None:
        path = root / f"d{d_nm:.3f}nm_blaze{blaze_deg:.3f}deg" / "theta_scans" / f"theta_scan_{int(round(energy_ev))}eV.csv"
    elif latest:
        scans = list(root.glob("*/theta_scans/theta_scan_*.csv"))
        path = max(scans, key=lambda item: item.stat().st_mtime) if scans else None
    else:
        for design in sorted(root.glob("d*nm_blaze*deg"), key=_natural_key):
            try:
                summary = pd.read_csv(design / "multilayer_theta_search_summary.csv").dropna(subset=["energy_ev", "selected_efficiency"])
                top = summary.loc[summary["selected_efficiency"].idxmax()]
                path = design / "theta_scans" / f"theta_scan_{int(round(float(top.energy_ev)))}eV.csv"
            except (OSError, ValueError, KeyError):
                scans = sorted((design / "theta_scans").glob("theta_scan_*.csv"), key=_natural_key)
                path = scans[0] if scans else None
            break
    if path is None or not path.is_file():
        return empty
    try:
        traces = _theta_traces(path.parent, legend=False, paths=[path])
    except (OSError, ValueError, KeyError):
        return empty  # the file may still be mid-write while the scan runs
    for trace in traces:
        for key in ("group", "role", "scan_class"):
            trace.pop(key, None)
        trace["showlegend"] = True
    if not traces:
        return empty
    d_value, profile = _design_values(path.parent.parent.relative_to(Path(study_dir)))
    energy = float(re.search(r"([0-9.]+)eV", path.name).group(1))
    layout = _base_layout(f"Theta scans: d = {d_value:.3f} nm, profile = {profile:.3f}, {energy:g} eV",
                          "Grazing angle (deg)", "Diffraction efficiency")
    return {"traces": traces, "layout": layout, "d_nm": d_value, "blaze_deg": profile, "energy_ev": energy}


def _workflow(summary_path: Path) -> dict[str, Any]:
    summary = pd.read_csv(summary_path).dropna(subset=["energy_ev", "selected_efficiency"])
    first = sorted((summary_path.parent / "theta_scans").glob("theta_scan_*.csv"), key=lambda p: float(re.search(r"([0-9.]+)eV", p.name).group(1)))[0]
    traces = [{"type": "scatter", "mode": "lines+markers", "x": _finite(summary.energy_ev), "y": _finite(summary.selected_efficiency), "name": "Final results", "xaxis": "x", "yaxis": "y"}]
    for trace in _theta_traces(first.parent, legend=False):
        if first.stem in trace["name"]:
            trace.update(xaxis="x2", yaxis="y2")
            traces.append(trace)
    layout = _base_layout("Theta-search workflow", "Energy (eV)", "Diffraction efficiency")
    layout.update({"grid": {"rows": 1, "columns": 2, "pattern": "independent"}, "xaxis2": {"title": "Grazing angle (deg)"}, "yaxis2": {"title": "Diffraction efficiency"}})
    return {"traces": traces, "layout": layout}


def _design_values(path: Path) -> tuple[float, float]:
    text = "/".join(path.parts)
    d_match = re.search(r"d([0-9.]+)nm", text)
    p_match = re.search(r"(?:blaze|depth)([0-9.]+)(?:deg|nm)", text)
    if not d_match or not p_match:
        raise ValueError("Could not determine d-spacing and profile parameter from the result folder.")
    return float(d_match.group(1)), float(p_match.group(1))


def _grating_plot(kind: str, directory: Path, root: Path, config: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    d_spacing, profile = _design_values(directory.relative_to(root))
    grating = MultilayerGratingDesigner(build_design_config(root, config))._build_grating(d_spacing, profile)
    if kind == "profile":
        from matplotlib.colors import to_hex

        x, heights = grating._tiled_profile_points(num_periods=3)
        x_grid, z_grid, material_map, labels = grating._material_plot_data(num_periods=3)
        colors = [to_hex(color) for color in grating._plot_material_colors(
            coating_stack=grating.resolved_stack(), material_labels=labels
        )]
        scale = []
        count = max(len(colors), 1)
        for index, color in enumerate(colors):
            scale.extend([[index / count, color], [(index + 1) / count, color]])
        traces = [
            {"type": "scatter", "mode": "lines", "x": _finite(x), "y": _finite(heights), "fill": "tozeroy", "name": "Grating groove", "xaxis": "x", "yaxis": "y"},
            {"type": "heatmap", "x": _finite(x_grid), "y": _finite(z_grid), "z": np.asarray(material_map).tolist(),
             "colorscale": scale, "zmin": -0.5, "zmax": len(labels) - 0.5,
             "colorbar": {"title": "Material", "tickvals": list(range(len(labels))), "ticktext": labels},
             "showscale": True, "name": "Material stack", "xaxis": "x2", "yaxis": "y2"},
        ]
        layout = _base_layout(f"{grating.__class__.__name__} Profile (Three Periods)", "x (nm)", "Depth (nm)")
        layout.update({"grid": {"rows": 2, "columns": 1, "pattern": "independent"}, "xaxis2": {"title": "x (nm)"}, "yaxis2": {"title": "z (nm)"}, "meta": {"material_labels": labels}})
        return {"traces": traces, "layout": layout}
    stack = grating.resolved_stack()
    layers, summary, repeat = stack._schematic_layers_and_summary()
    traces = [{"type": "scatter", "mode": "markers", "x": [0.5], "y": [0],
               "marker": {"opacity": 0}, "showlegend": False, "hoverinfo": "skip"}]
    shapes = []
    annotations = []
    top = 0.0
    material_colors: dict[str, str] = {}
    palette = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"]
    for index, (material, thickness) in enumerate(layers):
        label = getattr(material, "name", str(material))
        color = material_colors.setdefault(label, palette[len(material_colors) % len(palette)])
        bottom = top + float(thickness)
        shapes.append({"type": "rect", "x0": 0, "x1": 1, "y0": -bottom, "y1": -top,
                       "fillcolor": color, "line": {"color": "white"}})
        annotations.append({"x": 0.5, "y": -(top + bottom) / 2,
                            "text": f"{label} · {float(thickness):.3f} nm", "showarrow": False})
        top += float(thickness)
    layout = _base_layout(f"{stack.__class__.__name__} Schematic", "", "Depth from top surface (nm)")
    annotations.append({"text": "<br>".join(summary), "xref": "paper", "yref": "paper", "x": 1.02, "y": 1, "showarrow": False})
    layout.update({"showlegend": False, "shapes": shapes, "annotations": annotations,
                   "xaxis": {"visible": False, "range": [0, 1]}, "yaxis": {"title": "Depth from top surface (nm)", "range": [-top, 0]}})
    return {"traces": traces, "layout": layout}


def build_plot(descriptor: dict[str, Any], root: Path, config: dict[str, Any]) -> dict[str, Any]:
    kind, source = descriptor["kind"], descriptor["source"]
    if kind.startswith("survey_"):
        return _survey_plot(kind, source, config)
    if kind == "energy":
        return _energy_plot([source], config, False)
    if kind == "energy_comparison":
        return _energy_plot(sorted(source.glob("*/multilayer_theta_search_summary.csv")), config, True)
    if kind == "all_orders":
        return _all_orders(source)
    if kind == "workflow":
        return _workflow(source)
    if kind == "theta_all":
        return {"traces": _theta_traces(source), "layout": _base_layout("Interactive theta-search scans", "Grazing angle (deg)", "Diffraction efficiency")}
    if kind in {"profile", "stack"}:
        return _grating_plot(kind, source, root, config)
    raise ValueError(f"Unsupported diagnostic plot kind: {kind}")
