from pathlib import Path

import numpy as np
import pytest

from grax.web.afm_workflow import parse_afm_upload, process_afm_upload


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
