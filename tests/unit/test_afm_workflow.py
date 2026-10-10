import io
from pathlib import Path

import numpy as np
import pytest

from grax.web.afm_workflow import estimate_afm_period, parse_afm_upload, preview_afm_upload, process_afm_upload


def test_parse_afm_upload_accepts_header_csv_and_units() -> None:
    values = parse_afm_upload(b"x,z\n0,1\n1,2\n2,3\n", units="um")
    assert values.shape == (3, 2)


@pytest.mark.parametrize("raw", [b"", b"x\n1\n", b"a,b\n1,not-a-number\n"])
def test_parse_afm_upload_rejects_invalid_files(raw: bytes) -> None:
    with pytest.raises(ValueError):
        parse_afm_upload(raw)


def test_process_afm_upload_writes_profile_and_diagnostics(tmp_path: Path) -> None:
    x = np.linspace(0, 4000, 401)
    z = 20 * (1 - np.cos(2 * np.pi * x / 1000)) / 2
    raw = ("x,z\n" + "\n".join(f"{a},{b}" for a, b in zip(x, z))).encode()
    result = process_afm_upload(raw, tmp_path / "workflow", filename="scan.csv", units="nm", profile_type="blazed", period_nm=1000)
    assert result["points"] > 2
    assert (tmp_path / "workflow" / "profile.csv").is_file()
    assert result["diagnostics"]


def test_afm_preview_shows_raw_scan_and_checked_normalization_stages() -> None:
    raw = b"x,z\n0,5\n1,7\n2,8\n3,10\n"
    preview = preview_afm_upload(
        raw, units="um", profile_type="blazed", period_nm=None,
        reverse=True, zero_baseline=True,
    )

    assert preview["traces"]["raw"]["x"] == [0.0, 1000.0, 2000.0, 3000.0]
    assert preview["traces"]["reverse"]["z"] == [10000.0, 8000.0, 7000.0, 5000.0]
    assert preview["traces"]["baseline"]["z"] == [5000.0, 3000.0, 2000.0, 0.0]
    assert "troughs" not in preview["traces"]


def test_afm_preview_estimates_period_from_scan_without_locking_manual_value() -> None:
    x = np.linspace(0, 5000, 501)
    z = 20 * (1 - np.cos(2 * np.pi * x / 1000)) / 2
    raw = ("x,z\n" + "\n".join(f"{a},{b}" for a, b in zip(x, z))).encode()

    suggested = preview_afm_upload(raw, units="nm", profile_type="blazed", period_nm=None)
    corrected = preview_afm_upload(raw, units="nm", profile_type="blazed", period_nm=1050)

    assert suggested["estimated_period_nm"] == pytest.approx(1000, abs=10)
    assert "troughs" not in suggested["traces"]
    assert "period" in corrected["traces"]


def test_afm_period_estimate_requires_repeated_regular_features() -> None:
    assert estimate_afm_period(np.arange(100), np.ones(100)) is None
    assert estimate_afm_period(np.arange(7), np.arange(7)) is None


def test_afm_example_blazed_period_estimate_respects_units() -> None:
    example = Path(__file__).resolve().parents[2] / "examples/grating/data/afm_profile_example_blazed.txt"
    raw = example.read_bytes()

    meters = preview_afm_upload(raw, units="m", profile_type="blazed", period_nm=None)
    nanometers = preview_afm_upload(raw, units="nm", profile_type="blazed", period_nm=None)

    assert meters["estimated_period_nm"] == pytest.approx(1e6 / 600, rel=0.03)
    assert nanometers["estimated_period_nm"] == pytest.approx(meters["estimated_period_nm"] / 1e9)
    assert nanometers["suggested_units"] == "m"


def test_afm_example_blazed_matches_example_pipeline_and_previews_every_stage() -> None:
    example = Path(__file__).resolve().parents[2] / "examples/grating/data/afm_profile_example_blazed.txt"
    preview = preview_afm_upload(
        example.read_bytes(), units="m", profile_type="blazed", period_nm=1e6 / 600,
        reverse=True, zero_baseline=True, average=True, periodicity_ramp=True,
    )

    assert preview["stage_message"] == ""
    assert len(preview["traces"]["troughs"]["markers"]["x"]) == 5
    assert set(preview["traces"]) == {
        "raw", "reverse", "baseline", "troughs", "average", "ramp", "rescaled"
    }
    assert len(preview["traces"]["average"]["individuals"]) == 4
    assert preview["traces"]["rescaled"]["x"][-1] == pytest.approx(1e6 / 600)


def test_afm_preview_shows_extraction_average_and_ramp_only_when_selected() -> None:
    x = np.linspace(0, 4000, 401)
    z = 20 * (1 - np.cos(2 * np.pi * x / 1000)) / 2
    raw = ("x,z\n" + "\n".join(f"{a},{b}" for a, b in zip(x, z))).encode()
    selected = preview_afm_upload(raw, units="nm", profile_type="blazed", period_nm=1000)
    averaged = preview_afm_upload(
        raw, units="nm", profile_type="blazed", period_nm=1000,
        average=True, periodicity_ramp=True,
    )

    assert "troughs" in selected["traces"]
    assert "period" in selected["traces"]
    assert "selection" in selected["traces"]["troughs"]
    assert "rescaled" in selected["traces"]
    assert "average" not in selected["traces"]
    assert "ramp" not in selected["traces"]
    assert "average" in averaged["traces"]
    assert "period" not in averaged["traces"]
    assert "ramp" in averaged["traces"]
    assert "rescaled" in averaged["traces"]
    assert averaged["traces"]["rescaled"]["z"] == averaged["traces"]["ramp"]["z"]
    assert averaged["traces"]["ramp"]["before"] == averaged["traces"]["average"]


def test_afm_preview_retains_raw_scan_when_extraction_fails() -> None:
    preview = preview_afm_upload(
        b"x,z\n0,1\n1,1\n2,1\n3,1\n",
        units="nm", profile_type="blazed", period_nm=1000,
    )

    assert preview["traces"]["raw"]["z"] == [1.0, 1.0, 1.0, 1.0]
    assert "period" not in preview["traces"]
    assert "at least two troughs" in preview["stage_message"]


def test_afm_preview_downsamples_long_scans_without_losing_a_narrow_peak() -> None:
    x = np.arange(10000, dtype=float)
    z = np.zeros_like(x)
    z[4321] = 7.0
    raw = "\n".join(f"{a},{b}" for a, b in zip(x, z)).encode()

    preview = preview_afm_upload(raw, units="nm", profile_type="blazed", period_nm=None)
    curve = preview["traces"]["raw"]

    assert len(curve["x"]) <= 1600
    assert curve["x"][0] == 0
    assert curve["x"][-1] == 9999
    assert max(curve["z"]) == 7.0


def test_afm_preview_endpoint_handles_raw_scan_without_period_and_bad_upload(tmp_path: Path) -> None:
    from grax.web.app import create_app

    client = create_app(data_dir=tmp_path).test_client()
    raw = b"x,z\n0,1\n1,2\n2,3\n"
    response = client.post(
        "/gratings/afm/preview",
        data={"afm_file": (io.BytesIO(raw), "scan.csv"), "afm_units": "nm", "afm_reverse": "1"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert response.get_json()["traces"]["raw"]["z"] == [1.0, 2.0, 3.0]
    assert "reverse" in response.get_json()["traces"]
    assert response.get_json()["estimated_period_nm"] is None
    assert not (tmp_path / "afm_workflows").exists()

    invalid = client.post(
        "/gratings/afm/preview",
        data={"afm_file": (io.BytesIO(b"not a scan"), "bad.txt")},
        content_type="multipart/form-data",
    )
    assert invalid.status_code == 422
    assert invalid.get_json()["ok"] is False


def test_afm_preview_endpoint_returns_checked_later_stages(tmp_path: Path) -> None:
    from grax.web.app import create_app

    x = np.linspace(0, 4000, 401)
    z = 20 * (1 - np.cos(2 * np.pi * x / 1000)) / 2
    raw = ("x,z\n" + "\n".join(f"{a},{b}" for a, b in zip(x, z))).encode()
    response = create_app(data_dir=tmp_path).test_client().post(
        "/gratings/afm/preview",
        data={
            "afm_file": (io.BytesIO(raw), "scan.csv"),
            "afm_units": "nm",
            "afm_profile_type": "blazed",
            "afm_period_nm": "1000",
            "afm_average": "1",
            "afm_periodicity_ramp": "1",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    traces = response.get_json()["traces"]
    assert "raw" in traces
    assert "troughs" in traces
    assert "average" in traces
    assert "ramp" in traces
    assert not (tmp_path / "afm_workflows").exists()


def test_afm_form_has_inline_stage_previews(tmp_path: Path) -> None:
    from grax.web.app import create_app

    response = create_app(data_dir=tmp_path).test_client().get("/gratings/new")

    assert response.status_code == 200
    assert b"data-afm-preview-url" in response.data
    assert b"data-afm-target-period" in response.data
    assert b"data-afm-expected-period" not in response.data
    assert b"data-afm-groove-density" not in response.data
    assert b"data-afm-units-hint" in response.data
    assert b"data-afm-period-hint" in response.data
    assert b'data-afm-live-preview' in response.data
    assert b'name="afm_average" value="1" checked' in response.data
    assert response.data.index(b'name="afm_file"') < response.data.index(b'data-afm-plot="raw"')
    assert response.data.index(b'data-afm-plot="raw"') < response.data.index(b'name="afm_units"')
    for stage in (b"raw", b"troughs", b"period", b"average", b"reverse", b"baseline", b"ramp", b"rescaled"):
        assert b'data-afm-plot="' + stage + b'"' in response.data


def test_afm_live_preview_uses_rescaled_period_and_does_not_show_old_analytic_plot() -> None:
    source = Path(__file__).resolve().parents[2] / "src/grax/web/static"
    afm_js = (source / "afm.js").read_text(encoding="utf-8")
    web_js = (source / "web.js").read_text(encoding="utf-8")

    assert 'liveCurve = traces.rescaled || null' in afm_js
    assert 'renderCurve(liveFigure, repeatPeriod(liveCurve))' in afm_js
    assert 'image.classList.add("is-hidden")' in web_js
    assert '"Review the AFM period. Save will process the current settings."' in web_js


def test_processed_afm_profile_can_be_previewed_and_saved(tmp_path: Path) -> None:
    from grax.web.app import create_app
    from grax.web.persistence import GratingStore, build_grating_from_spec

    client = create_app(data_dir=tmp_path).test_client()
    x = np.linspace(0, 4000, 401)
    z = 20 * (1 - np.cos(2 * np.pi * x / 1000)) / 2
    raw = ("x,z\n" + "\n".join(f"{a},{b}" for a, b in zip(x, z))).encode()
    processed = client.post(
        "/gratings/afm/process",
        data={
            "afm_file": (io.BytesIO(raw), "scan.csv"), "afm_units": "nm",
            "afm_profile_type": "blazed", "afm_period_nm": "1000", "afm_average": "1",
        },
        content_type="multipart/form-data",
    )
    assert processed.status_code == 200
    profile = processed.get_json()
    assert profile["ok"] is True
    form = {
        "name": "AFM save test", "grating_type": "afm", "period_lpermm": "1000",
        "x_resolution_nm": "2", "z_resolution_nm": "0.5",
        "stack_type": "single_layer", "substrate_material": "Si",
        "layer_material": "Pt", "layer_thickness_nm": "20",
        "afm_profile_path": profile["profile_path"],
        "afm_source_filename": profile["source_filename"],
        "afm_units": profile["units"], "afm_profile_type": profile["profile_type"],
        "afm_period_nm": str(profile["period_nm"]),
    }

    preview = client.post("/_preview/grating", data=form)
    saved = client.post("/gratings", data=form)

    assert preview.get_json()["ok"] is True
    assert saved.status_code == 302
    spec = GratingStore(tmp_path / "saved_gratings").list()[0]
    assert spec["grating_type"] == "afm"
    assert build_grating_from_spec(spec).period_lpermm == 1000


def test_save_attempt_auto_processes_current_afm_settings_in_browser() -> None:
    source = Path(__file__).resolve().parents[2] / "src/grax/web/static"
    afm_js = (source / "afm.js").read_text(encoding="utf-8")
    web_js = (source / "web.js").read_text(encoding="utf-8")

    assert 'form.dispatchEvent(new Event("grax:afm-save-requested"))' in web_js
    assert 'gratingForm.addEventListener("grax:afm-save-requested"' in afm_js
    assert 'if (await processCurrentProfile()) gratingForm.requestSubmit()' in afm_js
    assert 'if (startedRevision !== settingsRevision)' in afm_js
