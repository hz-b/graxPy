"""Saved comparison axes, scan loops, and per-series legend overrides."""

import base64
import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest
from werkzeug.datastructures import MultiDict

from grax.web import app as web_app
from grax.web.app import create_app


def _write_sweep(root: Path, run_id: str = "sweep") -> None:
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True)
    scans = []
    for index, (order, coating) in enumerate(
        (order, coating) for order in (1, 2) for coating in ("Au", "Pt", "Ru")
    ):
        scans.append(
            {
                "id": f"scan-{index:06d}",
                "label": f"Diffraction order={order}; Coating={coating}",
                "parameters": {"diffraction_order": order, "stack.layer_material": coating},
                "axis_indices": {
                    "diffraction_order": (order - 1),
                    "stack.layer_material": index % 3,
                },
                "settings": {"diffraction_order": order},
            }
        )
    sweep = {
        "energy_values": [100.0, 110.0],
        "axes": [
            {
                "id": "diffraction_order",
                "label": "Diffraction order",
                "parameters": ["diffraction_order"],
            },
            {
                "id": "stack.layer_material",
                "label": "Coating",
                "parameters": ["stack.layer_material"],
            },
        ],
        "scans": scans,
    }
    manifest = {
        "id": run_id,
        "workflow": "energy_sweep",
        "display_name": run_id,
        "status": "completed",
        "run_input": {"diffraction_order": 1},
        "parameter_sweep": sweep,
        "artifacts": [],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with (run_dir / "all_orders.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scan_id", "energy_ev", "order", "efficiency"])
        writer.writeheader()
        for index, scan in enumerate(scans):
            for energy in (100.0, 110.0):
                if index == 5 and energy == 110.0:
                    continue
                for order in (-1, 0, 1, 2):
                    writer.writerow(
                        {
                            "scan_id": scan["id"],
                            "energy_ev": energy,
                            "order": order,
                            "efficiency": (index + 1) / 10,
                        }
                    )


def _form(*, loops=(), orders=(2,), x_value="energy", label="") -> MultiDict:
    fields = [
        ("run_ids", "sweep"),
        ("scan_selection_sweep", "1"),
        ("scans_sweep", "scan-000000"),
        ("x_axis_value", x_value),
    ]
    fields += [("orders_sweep", str(order)) for order in orders]
    fields += [("loops_sweep", loop) for loop in loops]
    if label:
        fields.append(("legend_label_sweep__scan-000001__order_2", label))
    return MultiDict(fields)


def _trace_values(trace: dict, axis: str) -> list[float]:
    encoded = trace[axis]
    if isinstance(encoded, list):
        return encoded
    return np.frombuffer(base64.b64decode(encoded["bdata"]), dtype=encoded["dtype"]).tolist()


def test_loop_expansion_axis_and_legend_persist_in_preview_and_save(
    tmp_path: Path, monkeypatch
) -> None:
    """Expand recorded combinations and keep axis and style in saved output."""
    _write_sweep(tmp_path)
    client = create_app(data_dir=tmp_path).test_client()
    page = client.get("/plots/new").get_data(as_text=True)
    assert 'name="x_axis_value"' in page
    assert "data-add-loop" in page
    assert "diffraction_order" in page
    assert "stack.layer_material" in page

    one = client.post("/_preview/plot", data=_form()).get_json()
    two = client.post("/_preview/plot", data=_form(loops=("diffraction_order",))).get_json()
    three = client.post("/_preview/plot", data=_form(loops=("stack.layer_material",))).get_json()
    form = _form(
        loops=("diffraction_order", "stack.layer_material"),
        x_value="energy_times_order",
        label="My Pt curve",
    )
    six = client.post("/_preview/plot", data=form).get_json()
    assert [len(payload["series_controls"]) for payload in (one, two, three, six)] == [1, 2, 3, 6]
    figure = json.loads(six["figure_json"])
    assert len({control["series_key"] for control in six["series_controls"]}) == 6
    assert figure["layout"]["xaxis"]["title"]["text"] == "Energy × Diffraction Order (eV)"
    assert _trace_values(figure["data"][0], "x") == [200.0, 220.0]
    assert math.isnan(_trace_values(figure["data"][5], "y")[1])
    assert figure["data"][1]["name"] == "My Pt curve"
    assert "Coating=Ru" in figure["data"][5]["name"]

    rendered = {}
    save_image = web_app._save_compare_image

    def capture_image(**kwargs):
        rendered.update(kwargs)
        save_image(**kwargs)

    monkeypatch.setattr(web_app, "_save_compare_image", capture_image)
    response = client.post("/plots", data=form)
    assert response.status_code == 302
    plot_id = response.headers["Location"].rsplit("/", 1)[-1]
    saved = json.loads((tmp_path / "plots" / plot_id / "manifest.json").read_text(encoding="utf-8"))
    assert saved["plot_config"]["x_axis_value"] == "energy_times_order"
    assert saved["plot_config"]["loop_selection"]["sweep"] == [
        "diffraction_order",
        "stack.layer_material",
    ]
    assert saved["plot_config"]["scan_selection"]["sweep"] == ["scan-000000"]
    assert (
        saved["plot_config"]["series_styles"]["sweep::scan::scan-000001::order::2"]["legend_label"]
        == "My Pt curve"
    )
    assert json.loads(saved["figure_json"])["data"][1]["name"] == "My Pt curve"
    assert (tmp_path / "plots" / plot_id / saved["plot_path"]).is_file()
    assert rendered["x_axis_value"] == "energy_times_order"
    assert rendered["series_controls"][1]["label"] == "My Pt curve"
    assert web_app._compare_x_values(
        rendered["selected_series"][0], rendered["x_axis_value"]
    ).tolist() == [200.0, 220.0]

    saved["plot_config"].pop("x_axis_value")
    saved.pop("figure_json")
    (tmp_path / "plots" / plot_id / "manifest.json").write_text(json.dumps(saved), encoding="utf-8")
    older_detail = client.get(response.headers["Location"])
    assert older_detail.status_code == 200
    assert b"Photon Energy (eV)" in older_detail.data


def test_loops_apply_per_run(tmp_path: Path) -> None:
    """Keep loop expansion local to its selected run."""
    _write_sweep(tmp_path)
    _write_sweep(tmp_path, run_id="second")
    form = _form(loops=("diffraction_order", "stack.layer_material"))
    form.add("run_ids", "second")
    form.add("orders_second", "2")
    form.add("scan_selection_second", "1")
    form.add("scans_second", "scan-000000")
    payload = (
        create_app(data_dir=tmp_path).test_client().post("/_preview/plot", data=form).get_json()
    )
    assert payload["ok"] is True
    assert len(payload["series_controls"]) == 7
    assert [len(run["scan_ids"]) for run in payload["selected_runs"]] == [6, 1]


@pytest.mark.parametrize(
    "order,expected", [(2, [200.0, 220.0]), (0, [0.0, 0.0]), (-1, [-100.0, -110.0])]
)
def test_energy_times_plotted_order_and_log_rejection(
    tmp_path: Path, order: int, expected: list[float]
) -> None:
    """Use the plotted order and reject invalid logarithmic values."""
    _write_sweep(tmp_path)
    client = create_app(data_dir=tmp_path).test_client()
    form = _form(orders=(order,), x_value="energy_times_order")
    preview = client.post("/_preview/plot", data=form).get_json()
    assert _trace_values(json.loads(preview["figure_json"])["data"][0], "x") == expected
    form.add("x_axis_type", "log")
    invalid = client.post("/_preview/plot", data=form).get_json()
    assert invalid["ok"] is (order > 0)
    if order <= 0:
        assert "positive" in invalid["error"]
        assert client.post("/plots", data=form).status_code == 400


def test_invalid_loop_parameter_is_rejected(tmp_path: Path) -> None:
    """Reject a loop that is absent from the saved sweep."""
    _write_sweep(tmp_path)
    form = _form(loops=("unknown",))
    response = (
        create_app(data_dir=tmp_path).test_client().post("/_preview/plot", data=form).get_json()
    )
    assert response["ok"] is False
    assert "loop parameter" in response["error"]
