"""Comparison plots: series loading, Plotly figures, saved images and exports."""

from __future__ import annotations

import base64
import csv
import io
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from . import parameter_scans
from .catalog import (
    _custom_grid_plot_definition,
    _is_plane_mirror_run,
    _plane_mirror_plot_definition,
    _run_selector_text,
)
from .common import (
    _is_safe_id,
    _plot_id,
)
from .run_files import (
    _available_orders,
    _load_order_series,
    _load_run_manifest,
)

matplotlib.use("Agg")

try:
    import plotly.graph_objects as go
    import plotly.io as plotly_io
    from plotly.offline import get_plotlyjs
    from plotly.subplots import make_subplots
except ModuleNotFoundError:  # pragma: no cover - optional dependency at runtime.
    go = None
    plotly_io = None
    get_plotlyjs = None
    make_subplots = None


def _require_plotly() -> None:
    """Require Plotly runtime dependencies for the Web UI."""

    if go is None or plotly_io is None or get_plotlyjs is None or make_subplots is None:
        raise RuntimeError(
            "Interactive Web UI plots require Plotly. Install them with either "
            "python -m pip install \"graxpy[web]\" or, from a local checkout, "
            "python -m pip install -e \".[web]\""
        )


def _plotly_bundle_text() -> str:
    """Return the bundled Plotly JavaScript source."""

    _require_plotly()
    assert get_plotlyjs is not None
    return str(get_plotlyjs())


PLOTLY_MARKER_SYMBOLS: tuple[tuple[str, str], ...] = (
    ("circle", "Circle"),
    ("square", "Square"),
    ("diamond", "Diamond"),
    ("triangle-up", "Triangle up"),
    ("triangle-down", "Triangle down"),
    ("cross", "Cross"),
    ("x", "X"),
    ("star", "Star"),
)


PLOTLY_DEFAULT_COLORS: tuple[str, ...] = (
    "#184e8b",
    "#bcd233",
    "#6cabe9",
    "#cb5a2e",
    "#6f8292",
    "#81768e",
    "#d9799c",
    "#093566",
    "#c9a95a",
    "#9ac6f3",
)


PLOTLY_LINE_STYLES: tuple[str, ...] = ("solid", "dot", "dash", "dashdot")


def _saved_plot_figure_json(data_dir: Path, manifest: dict[str, Any]) -> str | None:
    """Use the saved trace snapshot, or reconstruct an older plot when possible."""
    if manifest.get("figure_json"):
        return str(manifest["figure_json"])
    if not manifest.get("plot_config"):
        return None
    run_ids = [str(run["id"]) for run in manifest.get("selected_runs", [])]
    orders = {str(run["id"]): [int(value) for value in run.get("orders", [])]
              for run in manifest.get("selected_runs", [])}
    try:
        return _build_plotly_compare_payload(
            data_dir=data_dir, run_ids=run_ids, order_selection=orders,
            plot_options=dict(manifest["plot_config"]),
        )[1]
    except (ValueError, RuntimeError):
        return None


def _plot_export_zip(figure_json: str, *, include_script: bool) -> io.BytesIO:
    """Package one CSV per saved Plotly trace and an optional replay script."""
    assert plotly_io is not None
    figure = plotly_io.from_json(figure_json)
    specs = []
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, trace in enumerate(figure.data, start=1):
            filename = f"line-{index:03d}-{_plot_filename_stem(str(trace.name or 'series'))}.csv"
            csv_buffer = io.StringIO()
            writer = csv.writer(csv_buffer)
            writer.writerow(["x", "y"])
            x_values = _plotly_trace_values(trace.x)
            y_values = _plotly_trace_values(trace.y)
            for x, y in zip(x_values, y_values):
                writer.writerow([x, y])
            archive.writestr(filename, csv_buffer.getvalue())
            specs.append({
                "csv": filename, "label": str(trace.name or filename),
                "color": trace.line.color, "line_style": trace.line.dash,
                "marker": trace.marker.symbol, "marker_size": trace.marker.size,
            })
        if include_script:
            layout = figure.layout
            config = {
                "title": str(layout.title.text or ""),
                "x_label": str(layout.xaxis.title.text or "x"),
                "y_label": str(layout.yaxis.title.text or "y"),
                "x_scale": str(layout.xaxis.type or "linear"),
                "y_scale": str(layout.yaxis.type or "linear"),
                "lines": specs,
            }
            archive.writestr("plot.py", _plot_replay_script(config))
    output.seek(0)
    return output


def _plotly_trace_values(values: Any) -> list[Any]:
    """Decode Plotly's typed-array JSON encoding as well as ordinary arrays."""
    if isinstance(values, dict) and isinstance(values.get("bdata"), str):
        dtype = np.dtype(values.get("dtype", "f8"))
        decoded = np.frombuffer(base64.b64decode(values["bdata"]), dtype=dtype)
        shape = values.get("shape")
        if shape:
            decoded = decoded.reshape(tuple(int(value) for value in str(shape).split(",")))
        return decoded.tolist()
    if values is None:
        return []
    return np.asarray(values).tolist()


def _plot_replay_script(config: dict[str, Any]) -> str:
    """Return a simple, commented Matplotlib script that reads the CSV files."""
    marker_names = {
        "circle": "o", "square": "s", "diamond": "D", "triangle-up": "^",
        "triangle-down": "v", "cross": "+", "x": "x", "star": "*",
    }
    dash_names = {"solid": "-", "dot": ":", "dash": "--", "dashdot": "-."}
    lines = [
        "# This script uses the CSV files in the same folder.",
        "# Install the required packages with: python -m pip install pandas matplotlib",
        "from pathlib import Path",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "",
        "# Keep the CSV files and this script together in the same folder.",
        "data_folder = Path(__file__).resolve().parent",
        "",
        "# Read the plotted data. Each CSV file has an x column and a y column.",
    ]
    for index, item in enumerate(config["lines"], start=1):
        lines.append(f"data_{index} = pd.read_csv(data_folder / {item['csv']!r})")
    lines.extend(["", "# Draw one line for each CSV file."])
    for index, item in enumerate(config["lines"], start=1):
        args = [f"data_{index}['x']", f"data_{index}['y']"]
        options = [f"label={item['label']!r}", f"color={item['color']!r}",
                   f"linestyle={dash_names.get(item['line_style'], '-')!r}", "linewidth=1.6"]
        if item["marker_size"]:
            options.extend([f"marker={marker_names.get(item['marker'], 'o')!r}",
                            f"markersize={item['marker_size']!r}"])
        lines.append(f"plt.plot({', '.join(args + options)})")
    lines.extend([
        "", "# Add the title, axis labels, and legend.",
        f"plt.title({config['title']!r})",
        f"plt.xlabel({config['x_label']!r})",
        f"plt.ylabel({config['y_label']!r})",
    ])
    if config["x_scale"] == "log":
        lines.append('plt.xscale("log")')
    if config["y_scale"] == "log":
        lines.append('plt.yscale("log")')
    lines.extend([
        "plt.grid(True)", "plt.legend()", "",
        "# Save the image next to this script, then display the plot.",
        "plt.savefig(data_folder / 'my_plot.png', dpi=150, bbox_inches='tight')",
        "plt.show()", "",
    ])
    return "\n".join(lines)


def _plot_selected_run_entries(data_dir: Path, plot: dict[str, Any]) -> list[dict[str, Any]]:
    """Annotate saved-plot run references without changing their manifest."""

    entries: list[dict[str, Any]] = []
    for selected in plot.get("selected_runs", []):
        entry = dict(selected)
        run_id = str(entry.get("id", ""))
        entry["id"] = run_id
        run_manifest = _load_run_manifest(data_dir, run_id)
        entry["available"] = run_manifest is not None
        entry["is_plane_mirror"] = bool(run_manifest and _is_plane_mirror_run(run_manifest))
        entry["url"] = (
            f"/plane-mirror-runs/{run_id}" if entry["is_plane_mirror"]
            else f"/runs/{run_id}" if entry["available"] else None
        )
        entries.append(entry)
    return entries


def _build_plot_preview(*, data_dir: Path, form_data: Any) -> dict[str, Any]:
    """Build a live combined-plot preview from saved runs and selected orders."""
    run_ids = [str(run_id) for run_id in form_data.getlist("run_ids")]
    if not run_ids:
        return {
            "ok": False,
            "error": "Select at least one saved run.",
            "figure_json": None,
            "selected_runs": [],
            "series_controls": [],
        }

    order_selection: dict[str, list[int]] = {}
    for run_id in run_ids:
        orders = [int(value) for value in form_data.getlist(f"orders_{run_id}") if str(value).strip() != ""]
        manifest = _load_run_manifest(data_dir, run_id)
        if manifest and _is_plane_mirror_run(manifest):
            order_selection[run_id] = [0]
            continue
        if not orders:
            return {
                "ok": False,
                "error": f"Select at least one diffraction order for run {run_id}.",
                "figure_json": None,
                "selected_runs": [],
                "series_controls": [],
            }
        order_selection[run_id] = orders

    plot_options = _plot_options_from_form(
        form_data=form_data,
        run_ids=run_ids,
        order_selection=order_selection,
        data_dir=data_dir,
    )
    try:
        selected_runs, figure_json, series_controls = _build_plotly_compare_payload(
            data_dir=data_dir,
            run_ids=run_ids,
            order_selection=order_selection,
            plot_options=plot_options,
        )
    except ValueError as error:
        return {
            "ok": False,
            "error": str(error),
            "figure_json": None,
            "selected_runs": [],
            "series_controls": [],
        }

    return {
        "ok": True,
        "error": "",
        "figure_json": figure_json,
        "selected_runs": selected_runs,
        "series_controls": series_controls,
        "plot_options": plot_options,
    }


def _build_combined_plot(
    *,
    data_dir: Path,
    run_ids: list[str],
    order_selection: dict[str, list[int]],
    title: str,
    plot_options: dict[str, Any],
    existing_plot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and save a combined plot for several runs and selected orders."""
    run_summaries, figure_json, series_controls = _build_plotly_compare_payload(
        data_dir=data_dir,
        run_ids=run_ids,
        order_selection=order_selection,
        plot_options=plot_options,
    )
    _, selected_series = _load_selected_plot_series(
        data_dir=data_dir, run_ids=run_ids, order_selection=order_selection,
        scan_selection=plot_options.get("scan_selection", {}),
        loop_selection=plot_options.get("loop_selection", {}),
    )
    plot_root = data_dir / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    plot_id = str(existing_plot["id"]) if existing_plot else _plot_id()
    plot_dir = plot_root / plot_id
    plot_dir.mkdir(parents=True, exist_ok=True)
    name = str(plot_options.get("name") or title)
    image_path = plot_dir / (str(existing_plot["plot_path"]) if existing_plot else f"{_plot_filename_stem(name)}.png")
    _save_compare_image(
        selected_series=selected_series,
        series_controls=series_controls,
        title=title,
        x_axis_value=plot_options["x_axis_value"],
        x_axis_type=plot_options["x_axis_type"],
        y_axis_type=plot_options["y_axis_type"],
        output_path=image_path,
    )
    if existing_plot:
        for old_copy in existing_plot.get("run_images", []):
            (data_dir / "runs" / str(old_copy["run_id"]) / str(old_copy["filename"])).unlink(missing_ok=True)
    run_images = []
    for run in run_summaries:
        run_dir = data_dir / "runs" / str(run["id"])
        destination = _unused_plot_image_path(run_dir, image_path.name)
        shutil.copyfile(image_path, destination)
        run_images.append({"run_id": run["id"], "filename": destination.name})

    manifest = {
        "id": plot_id,
        "created_at": existing_plot["created_at"] if existing_plot else datetime.now().isoformat(timespec="seconds"),
        "title": title,
        "name": name,
        "plot_path": image_path.name,
        "run_images": run_images,
        "selected_runs": run_summaries,
        "plot_config": {
            "title": title,
            "x_axis_value": plot_options["x_axis_value"],
            "x_axis_type": plot_options["x_axis_type"],
            "y_axis_type": plot_options["y_axis_type"],
            "scan_selection": plot_options.get("scan_selection", {}),
            "loop_selection": plot_options.get("loop_selection", {}),
            "series_styles": {
                item["series_key"]: {
                    "legend_label": item["legend_label"],
                    "color": item["color"],
                    "marker_symbol": item["marker_symbol"],
                    "marker_size": item["marker_size"],
                    "line_style": item["line_style"],
                }
                for item in series_controls
            },
        },
        "figure_json": figure_json,
    }
    with (plot_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest


def _plot_filename_stem(title: str) -> str:
    """Return a useful, safe filename based on the plot title."""

    stem = "".join(character if character.isalnum() or character in "-_" else "_" for character in title.strip())
    return stem.strip("_-")[:80] or "plot"


def _unused_plot_image_path(directory: Path, filename: str) -> Path:
    """Choose a path beside a run manifest without overwriting a prior view."""

    candidate = directory / filename
    suffix = 2
    while candidate.exists():
        candidate = directory / f"{Path(filename).stem}_{suffix}.png"
        suffix += 1
    return candidate


def _save_compare_image(
    *, selected_series: list[dict[str, Any]], series_controls: list[dict[str, Any]],
    title: str, x_axis_value: str, x_axis_type: str, y_axis_type: str, output_path: Path,
) -> None:
    """Render the styled comparison as a PNG for run-folder access."""

    style_by_key = {style["series_key"]: style for style in series_controls}
    line_styles = {"solid": "-", "dot": ":", "dash": "--", "dashdot": "-."}
    markers = {"circle": "o", "square": "s", "diamond": "D", "triangle-up": "^",
               "triangle-down": "v", "cross": "+", "x": "x", "star": "*"}
    figure, axis = plt.subplots(figsize=(12, 6))
    try:
        for series in selected_series:
            style = style_by_key[series["series_key"]]
            axis.plot(
                _compare_x_values(series, x_axis_value),
                np.asarray(series["efficiencies"], dtype=float),
                label=style["label"], color=style["color"],
                linestyle=line_styles[style["line_style"]],
                marker=markers.get(style["marker_symbol"], "o"),
                markersize=style["marker_size"], linewidth=1.6,
            )
        axis.set_xscale(x_axis_type)
        axis.set_yscale(y_axis_type)
        axis.set_xlabel("Grazing angle (deg)" if selected_series[0].get("x_axis_id") == "grazing_angle_deg" else _compare_x_label(x_axis_value))
        axis.set_ylabel("Diffraction Efficiency")
        axis.set_title(title, pad=16)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=min(3, len(selected_series)))
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(figure)


def _plot_options_from_form(
    *,
    form_data: Any,
    run_ids: list[str],
    order_selection: dict[str, list[int]],
    data_dir: Path,
) -> dict[str, Any]:
    """Return normalized compare-plot configuration from form data."""

    plot_options = {
        "title": str(form_data.get("title", "")).strip() or "Combined plot",
        "name": str(form_data.get("plot_name", "")).strip(),
        "x_axis_value": _normalize_compare_x_value(form_data.get("x_axis_value")),
        "x_axis_type": _normalize_axis_type(form_data.get("x_axis_type")),
        "y_axis_type": _normalize_axis_type(form_data.get("y_axis_type")),
        "series_styles": {},
        "scan_selection": {},
        "loop_selection": {},
    }
    for run_id in run_ids:
        scan_ids = form_data.getlist(f"scans_{run_id}")
        if f"scan_selection_{run_id}" in form_data:
            plot_options["scan_selection"][run_id] = scan_ids
        plot_options["loop_selection"][run_id] = list(dict.fromkeys(form_data.getlist(f"loops_{run_id}")))
        manifest = _load_run_manifest(data_dir, run_id)
        run_dir = data_dir / "runs" / run_id
        sweep = (
            (manifest or {}).get("parameter_sweep")
            or _custom_grid_plot_definition(manifest or {})
            or (_plane_mirror_plot_definition(manifest, run_dir) if manifest and _is_plane_mirror_run(manifest) else None)
        )
        all_scan_ids = [scan["id"] for scan in (sweep or {}).get("scans", [])]
        for order in order_selection.get(run_id, []):
            for scan_id in (all_scan_ids if all_scan_ids else [None]):
                token = _series_token(run_id, int(order), scan_id)
                series_key = _series_key(run_id, int(order), scan_id)
                marker_symbol = str(form_data.get(f"marker_symbol_{token}", "")).strip() or "circle"
                valid_marker_symbols = {value for value, _ in PLOTLY_MARKER_SYMBOLS}
                if marker_symbol not in valid_marker_symbols:
                    marker_symbol = "circle"
                try:
                    raw_marker_size = form_data.get(f"marker_size_{token}", 3)
                    marker_size = int(float(raw_marker_size if raw_marker_size not in (None, "") else 3))
                except (TypeError, ValueError):
                    marker_size = 3
                marker_size = max(0, min(marker_size, 24))
                legend_label = str(form_data.get(f"legend_label_{token}", "")).strip()
                plot_options["series_styles"][series_key] = {
                    "legend_label": legend_label,
                    "color": _normalize_hex_color(form_data.get(f"color_{token}")),
                    "marker_symbol": marker_symbol,
                    "marker_size": marker_size,
                    "line_style": str(form_data.get(f"line_style_{token}", "")).strip(),
                }
    return plot_options


def _normalize_compare_x_value(raw_value: Any) -> str:
    """Default older saved plots to photon energy."""
    return "energy_times_order" if str(raw_value or "") == "energy_times_order" else "energy"


def _compare_x_label(x_axis_value: str) -> str:
    return "Energy × Diffraction Order (eV)" if x_axis_value == "energy_times_order" else "Photon Energy (eV)"


def _compare_x_values(series: dict[str, Any], x_axis_value: str) -> np.ndarray:
    energies = np.asarray(series["energies"], dtype=float)
    if series.get("is_plane_mirror") or series.get("x_axis_id") == "grazing_angle_deg":
        return energies
    return energies * int(series["order"]) if x_axis_value == "energy_times_order" else energies


def _normalize_axis_type(raw_value: Any) -> str:
    """Return a safe Plotly axis type."""

    value = str(raw_value or "").strip().lower()
    if value == "log":
        return "log"
    return "linear"


def _normalize_hex_color(raw_value: Any) -> str | None:
    """Return a normalized hex color or None when invalid."""

    value = str(raw_value or "").strip()
    if len(value) == 7 and value.startswith("#"):
        hex_digits = value[1:]
        if all(character in "0123456789abcdefABCDEF" for character in hex_digits):
            return value.lower()
    return None


def _series_key(run_id: str, order: int, scan_id: str | None = None) -> str:
    """Return the stable manifest key for one compare-plot series."""

    return f"{run_id}::scan::{scan_id}::order::{int(order)}" if scan_id else f"{run_id}::order::{int(order)}"


def _series_token(run_id: str, order: int, scan_id: str | None = None) -> str:
    """Return the form-safe token for one compare-plot series."""

    safe_run_id = "".join(character if character.isalnum() else "_" for character in run_id)
    return f"{safe_run_id}__{scan_id}__order_{int(order)}" if scan_id else f"{safe_run_id}__order_{int(order)}"


def _default_series_style(index: int) -> dict[str, Any]:
    """Return the default style for one compare-plot series."""

    marker_cycle = [value for value, _ in PLOTLY_MARKER_SYMBOLS]
    return {
        "color": PLOTLY_DEFAULT_COLORS[index % len(PLOTLY_DEFAULT_COLORS)],
        "marker_symbol": marker_cycle[index % len(marker_cycle)],
        "marker_size": 3,
        "line_style": PLOTLY_LINE_STYLES[index % len(PLOTLY_LINE_STYLES)],
    }


def _build_plotly_compare_payload(
    *,
    data_dir: Path,
    run_ids: list[str],
    order_selection: dict[str, list[int]],
    plot_options: dict[str, Any],
) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]]]:
    """Return selected-run summaries, serialized Plotly figure, and style controls."""

    _require_plotly()
    run_summaries, selected_series = _load_selected_plot_series(
        data_dir=data_dir,
        run_ids=run_ids,
        order_selection=order_selection,
        scan_selection=plot_options.get("scan_selection", {}),
        loop_selection=plot_options.get("loop_selection", {}),
    )
    if not selected_series:
        raise ValueError("Select at least one scan or mirror reflectivity series.")
    axis_ids = {series.get("x_axis_id", "energy_ev") for series in selected_series}
    if len(axis_ids) > 1:
        raise ValueError("Compare angle sweeps separately from energy scans; their horizontal axes differ.")
    x_axis_value = _normalize_compare_x_value(plot_options.get("x_axis_value"))
    if x_axis_value == "energy_times_order" and any(series.get("is_plane_mirror") for series in selected_series):
        raise ValueError("Energy × diffraction order is only available for grating runs. Choose Energy for mirror reflectivity.")
    series_controls = _series_controls_for_selected_series(
        selected_series=selected_series,
        saved_styles=dict(plot_options.get("series_styles", {})),
    )
    if plot_options["x_axis_type"] == "log" and any(
        np.any(_compare_x_values(series, x_axis_value) <= 0) for series in selected_series
    ):
        raise ValueError("Energy × diffraction order must be positive for a logarithmic X axis.")
    figure = _build_plotly_compare_figure(
        selected_series=selected_series,
        title=str(plot_options["title"]),
        x_axis_value=x_axis_value,
        x_axis_type=str(plot_options["x_axis_type"]),
        y_axis_type=str(plot_options["y_axis_type"]),
        series_controls=series_controls,
    )
    assert plotly_io is not None
    return run_summaries, plotly_io.to_json(figure, validate=False), series_controls


def _load_selected_plot_series(
    *,
    data_dir: Path,
    run_ids: list[str],
    order_selection: dict[str, list[int]],
    scan_selection: dict[str, list[str]] | None = None,
    loop_selection: dict[str, list[str]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return selected run summaries plus all order series requested for comparison."""

    run_summaries: list[dict[str, Any]] = []
    selected_series: list[dict[str, Any]] = []
    for run_id in run_ids:
        run_dir = data_dir / "runs" / run_id
        manifest_path = run_dir / "manifest.json"
        if not _is_safe_id(run_id) or not manifest_path.exists():
            continue
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        run_label = _run_selector_text(manifest)
        legend_name = str(manifest.get("display_name") or manifest.get("grating_name") or run_id)
        if _is_plane_mirror_run(manifest):
            sweep = _plane_mirror_plot_definition(manifest, run_dir)
            if not sweep:
                continue
            run_summaries.append({"id": run_id, "name": run_label, "orders": [],
                                  "is_plane_mirror": True})
            chosen = (scan_selection or {}).get(run_id)
            scan_ids = _expanded_plot_scan_ids(sweep, chosen, (loop_selection or {}).get(run_id, []))
            run_summaries[-1]["scan_ids"] = scan_ids
            scan_definitions = {scan["id"]: scan for scan in sweep["scans"]}
            results_by_id = {
                f"mirror-case-{index:06d}": result
                for index, result in enumerate(sweep["mirror_results"])
            }
            for scan_id in scan_ids:
                result = results_by_id.get(scan_id)
                if not result:
                    continue
                scan = scan_definitions[scan_id]
                mode = str(result.get("mode") or "energy_sweep")
                series = {
                    "label": f"{legend_name} · {scan['label']}",
                    "order": 0,
                    "scan_id": scan_id,
                    "energies": result.get("x", []),
                    "x_axis_id": "grazing_angle_deg" if mode == "angle_sweep" else "energy_ev",
                    "efficiencies": result.get("reflectivity", []),
                    "is_plane_mirror": True,
                }
                series["series_key"] = _series_key(run_id, 0, scan_id)
                series["series_token"] = _series_token(run_id, 0, scan_id)
                series["run_id"] = run_id
                selected_series.append(series)
            continue
        orders = order_selection.get(run_id) or _available_orders(run_dir)
        run_summaries.append({"id": run_id, "name": run_label, "orders": orders})
        sweep = manifest.get("parameter_sweep") or _custom_grid_plot_definition(manifest)
        custom_grid_plot = bool(sweep and not manifest.get("parameter_sweep"))
        if sweep:
            chosen = (scan_selection or {}).get(run_id)
            scan_ids = _expanded_plot_scan_ids(sweep, chosen, (loop_selection or {}).get(run_id, []))
            run_summaries[-1]["scan_ids"] = scan_ids
        else:
            scan_ids = [None]
        scan_definitions = {scan["id"]: scan for scan in sweep["scans"]} if sweep else {}
        for order in orders:
            scan_values = (
                _custom_grid_plot_series(run_dir, sweep, int(order)) if custom_grid_plot
                else parameter_scans.load_all_series(run_dir, sweep, order) if sweep else {}
            )
            for scan_id in scan_ids:
                if sweep:
                    scan = scan_definitions[scan_id]
                    series = {
                        "label": f"{legend_name} · {scan['label']}",
                        "order": int(order),
                        "scan_id": scan_id,
                        "energies": sweep["energy_values"] if custom_grid_plot else parameter_scans.sample_values(sweep),
                        "x_axis_id": sweep.get("sample_axis", {}).get("id", "energy_ev"),
                        "efficiencies": scan_values[scan_id],
                    }
                else:
                    series = _load_order_series(run_dir, order=order, label=legend_name)
                if series is not None:
                    series["series_key"] = _series_key(run_id, int(order), scan_id)
                    series["series_token"] = _series_token(run_id, int(order), scan_id)
                    series["run_id"] = run_id
                    selected_series.append(series)
    return run_summaries, selected_series


def _custom_grid_plot_series(run_dir: Path, sweep: dict[str, Any], order: int) -> dict[str, list[float]]:
    """Keep Custom grid energy cases separate by their saved parameter values."""
    energies = sweep["energy_values"]
    energy_index = {value: index for index, value in enumerate(energies)}
    values = {scan["id"]: [float("nan")] * len(energies) for scan in sweep["scans"]}
    path = run_dir / "all_orders.csv"
    if not path.exists():
        return values
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                if abs(int(float(row["order"]))) != abs(order):
                    continue
                scan_id, energy = sweep["case_map"][row["case_id"]]
                values[scan_id][energy_index[energy]] = float(row["efficiency"])
            except (KeyError, TypeError, ValueError):
                continue
    return values


def _expanded_plot_scan_ids(
    sweep: dict[str, Any], chosen: list[str] | None, loops: list[str]
) -> list[str]:
    """Expand seed scans over recorded values of the chosen parameters."""
    scans = sweep["scans"]
    available = {
        parameter for axis in sweep.get("axes", []) for parameter in axis.get("parameters", [])
    }
    if len(loops) != len(set(loops)) or any(parameter not in available for parameter in loops):
        raise ValueError("Choose each available loop parameter only once.")
    # A run and order are sufficient to create a plot. An empty explicit scan
    # selection can occur when every grouped/custom-grid checkbox is cleared;
    # treat that as "show all scans" instead of producing zero series and a
    # misleading run/order validation error.
    available_scan_ids = {scan["id"] for scan in scans}
    chosen_ids = (set(chosen) & available_scan_ids) if chosen else available_scan_ids
    if not chosen_ids:
        chosen_ids = available_scan_ids
    seeds = [scan for scan in scans if scan["id"] in chosen_ids]
    if not loops:
        return [scan["id"] for scan in seeds]
    fixed_parameters = sorted(available - set(loops))

    def fixed_key(scan: dict[str, Any]) -> tuple[str, ...]:
        return tuple(
            json.dumps(scan["parameters"].get(parameter), sort_keys=True, ensure_ascii=False)
            for parameter in fixed_parameters
        )

    seed_keys = {fixed_key(scan) for scan in seeds}
    return [scan["id"] for scan in scans if fixed_key(scan) in seed_keys]


def _series_controls_for_selected_series(
    *,
    selected_series: list[dict[str, Any]],
    saved_styles: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the normalized per-series style controls for one compare plot."""

    series_controls: list[dict[str, Any]] = []
    for index, series in enumerate(selected_series):
        default_style = _default_series_style(index)
        saved_style = dict(saved_styles.get(str(series["series_key"]), {}))
        marker_symbol = str(saved_style.get("marker_symbol", default_style["marker_symbol"]))
        if marker_symbol not in {value for value, _ in PLOTLY_MARKER_SYMBOLS}:
            marker_symbol = str(default_style["marker_symbol"])
        try:
            marker_size = int(float(saved_style.get("marker_size", default_style["marker_size"])))
        except (TypeError, ValueError):
            marker_size = int(default_style["marker_size"])
        marker_size = max(0, min(marker_size, 24))
        line_style = str(saved_style.get("line_style", "solid" if saved_style else default_style["line_style"]))
        if line_style not in PLOTLY_LINE_STYLES:
            line_style = str(default_style["line_style"])
        default_label = (
            str(series["label"])
            if series.get("is_plane_mirror")
            else f"{series['label']} · order: {series['order']}"
        )
        legend_label = str(saved_style.get("legend_label", "")).strip()
        series_controls.append(
            {
                "series_key": str(series["series_key"]),
                "series_token": str(series["series_token"]),
                "run_id": str(series["run_id"]),
                "order": int(series["order"]),
                "is_plane_mirror": bool(series.get("is_plane_mirror")),
                "label": legend_label or default_label,
                "legend_label": legend_label,
                "color": _normalize_hex_color(saved_style.get("color")) or str(default_style["color"]),
                "marker_symbol": marker_symbol,
                "marker_size": marker_size,
                "line_style": line_style,
            }
        )
    return series_controls


def _build_plotly_compare_figure(
    *,
    selected_series: list[dict[str, Any]],
    title: str,
    x_axis_value: str,
    x_axis_type: str,
    y_axis_type: str,
    series_controls: list[dict[str, Any]],
) -> Any:
    """Build the interactive Plotly figure for one compare plot."""

    _require_plotly()
    assert go is not None
    control_map = {item["series_key"]: item for item in series_controls}
    x_label = "Grazing angle (deg)" if selected_series[0].get("x_axis_id") == "grazing_angle_deg" else _compare_x_label(x_axis_value)
    figure = go.Figure()
    for series in selected_series:
        style = control_map[str(series["series_key"])]
        figure.add_trace(
            go.Scatter(
                x=_compare_x_values(series, x_axis_value),
                y=np.asarray(series["efficiencies"], dtype=float),
                mode="lines+markers" if style["marker_size"] > 0 else "lines",
                name=style["label"],
                line={"color": style["color"], "width": 1.6, "dash": style["line_style"]},
                marker={
                    "color": style["color"],
                    "symbol": style["marker_symbol"],
                    "size": style["marker_size"],
                },
                hovertemplate=f"{x_label}: %{{x:.3f}}<br>Efficiency: %{{y:.6f}}<extra>%{{fullData.name}}</extra>",
            )
        )
    figure.update_layout(
        title=title,
        template="plotly_white",
        showlegend=True,
        legend={"orientation": "h", "yanchor": "top", "y": -0.25, "x": 0.0},
        margin={"l": 64, "r": 24, "t": 82, "b": 140},
    )
    figure.update_xaxes(title_text=x_label, type=x_axis_type, showgrid=True, zeroline=False)
    y_label = (
        "Reflectivity" if all(series.get("is_plane_mirror") for series in selected_series)
        else "Efficiency / reflectivity" if any(series.get("is_plane_mirror") for series in selected_series)
        else "Diffraction Efficiency"
    )
    figure.update_yaxes(title_text=y_label, type=y_axis_type, showgrid=True, zeroline=False)
    return figure


def _render_combined_plot_image(
    *,
    data_dir: Path,
    run_ids: list[str],
    order_selection: dict[str, list[int]],
    title: str,
    output_path: Path,
) -> list[dict[str, Any]]:
    """Render a combined plot image and return the selected run summary."""
    run_summaries: list[dict[str, Any]] = []
    selected_series: list[dict[str, Any]] = []
    for run_id in run_ids:
        run_dir = data_dir / "runs" / run_id
        manifest_path = run_dir / "manifest.json"
        if not _is_safe_id(run_id) or not manifest_path.exists():
            continue
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        run_label = str(manifest.get("display_name") or manifest.get("grating_name", run_id))
        orders = order_selection.get(run_id) or _available_orders(run_dir)
        run_summaries.append({"id": run_id, "name": run_label, "orders": orders})
        for order in orders:
            series = _load_order_series(run_dir, order=order, label=run_label)
            if series is not None:
                selected_series.append(series)

    if not selected_series:
        raise ValueError("Select at least one run and one diffraction order.")

    figure, axis = plt.subplots(figsize=(10, 6))
    markers = ["o", "s", "^", "d", "v", "x", "*"]
    for index, series in enumerate(selected_series):
        axis.plot(
            np.asarray(series["energies"], dtype=float),
            np.asarray(series["efficiencies"], dtype=float),
            marker=markers[index % len(markers)],
            linewidth=1.2,
            label=f"{series['label']} · Order {series['order']}",
        )
    axis.set_xlabel("Photon Energy (eV)")
    axis.set_ylabel("Diffraction Efficiency")
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return run_summaries
