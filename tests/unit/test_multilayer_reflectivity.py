"""Unit tests for :mod:`grax.multilayer_reflectivity` with a stubbed XRT engine."""

from __future__ import annotations

import numpy as np
import pytest

from grax import multilayer_reflectivity as mr
from grax.multilayer_reflectivity import MultilayerReflectivity

pytestmark = pytest.mark.unit

_BRAGG_DEG = 1.0
_PEAK_OFFSET_DEG = 0.05
_PEAK_WIDTH_DEG = 0.03


class _FakeStack:
    """Synthetic multilayer with a Gaussian Bragg peak just above ``_BRAGG_DEG``."""

    def __init__(self, **_kwargs: object) -> None:
        self.kwargs = _kwargs

    def get_Bragg_angle(self, energy_ev: float, order: int) -> float:  # noqa: N802 - XRT name
        """Return a fixed Bragg grazing angle in radians."""

        return float(np.deg2rad(_BRAGG_DEG))

    def get_amplitude(
        self, energy_ev: float, sin_theta: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(rs, rp)`` amplitudes for a Gaussian reflectivity bump."""

        theta_deg = np.rad2deg(np.arcsin(np.clip(sin_theta, -1.0, 1.0)))
        center = _BRAGG_DEG + _PEAK_OFFSET_DEG
        reflectivity = 0.8 * np.exp(-0.5 * ((theta_deg - center) / _PEAK_WIDTH_DEG) ** 2)
        rs = np.sqrt(reflectivity)
        rp = np.sqrt(0.25 * reflectivity)
        return rs, rp


class _FakeMaterials:
    """Stand-in for ``xrt.backends.raycing.materials``."""

    @staticmethod
    def Material(name: str, rho: float | None = None) -> dict[str, object]:  # noqa: N802 - XRT name
        """Return a lightweight material record."""

        return {"name": name, "rho": rho}

    @staticmethod
    def Multilayer(**kwargs: object) -> _FakeStack:  # noqa: N802 - XRT name
        """Return the synthetic stack."""

        return _FakeStack(**kwargs)


@pytest.fixture
def fake_xrt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install the fake XRT materials module."""

    monkeypatch.setattr(mr, "_xrt_materials", lambda: _FakeMaterials)


def test_reflectivity_vs_energy_columns_and_peak(fake_xrt: None) -> None:
    """The frame has the documented columns and the peak sits on the bump."""

    engine = MultilayerReflectivity(("Ru", 12.1), 1.35, ("C", 2.52), 1.35, 40)
    frame = engine.reflectivity_vs_energy([8000.0, 9000.0, 10000.0], bragg_order=1)

    assert list(frame.columns) == list(mr._COLUMNS)
    assert len(frame) == 3
    # s-pol reflectivity peaks higher than p-pol for this synthetic stack.
    assert (frame["peak_rs"] > frame["peak_rp"]).all()
    # The selected angle is the Gaussian center, at or above the Bragg angle.
    assert np.allclose(frame["peak_angle_deg"], _BRAGG_DEG + _PEAK_OFFSET_DEG, atol=2.0e-3)
    assert (frame["peak_angle_deg"] >= frame["bragg_angle_deg"]).all()
    assert np.isfinite(frame["fwhm_deg"]).all()
    assert np.isfinite(frame["fwhm_ev"]).all()
    # Analytic FWHM of a Gaussian is 2*sqrt(2 ln 2) * sigma.
    assert np.allclose(frame["fwhm_deg"], 2.3548 * _PEAK_WIDTH_DEG, atol=5.0e-3)


def test_reflectivity_vs_energy_writes_results_csv(fake_xrt: None, tmp_path) -> None:
    """``save_recap`` writes ``results.csv`` next to the requested directory."""

    engine = MultilayerReflectivity(
        ("Ru", 12.1), 1.35, ("C", 2.52), 1.35, 40, save_recap=tmp_path / "d_2.7nm"
    )
    engine.reflectivity_vs_energy([9000.0], bragg_order=1)
    assert (tmp_path / "d_2.7nm" / "results.csv").is_file()


def test_individuals_plots_written(fake_xrt: None, tmp_path) -> None:
    """``individuals`` writes one PNG per energy."""

    engine = MultilayerReflectivity(
        ("Ru", 12.1), 1.35, ("C", 2.52), 1.35, 40, save_recap=tmp_path, individuals=True
    )
    engine.reflectivity_vs_energy([8000.0, 9000.0], bragg_order=1)
    pngs = sorted((tmp_path / "individuals").glob("*.png"))
    assert [p.name for p in pngs] == ["8000_eV.png", "9000_eV.png"]


def test_material_spec_accepted(fake_xrt: None) -> None:
    """A :class:`grax.MaterialSpec` with a density is unpacked like a pair."""

    from grax import MaterialSpec

    engine = MultilayerReflectivity(
        MaterialSpec("Ru", 12.1), 1.35, MaterialSpec("C", 2.52), 1.35, 40
    )
    assert engine.material_a_name == "Ru"
    assert engine.density_b == pytest.approx(2.52)


def test_constructor_rejects_bad_material() -> None:
    """A material without a density is rejected."""

    from grax import MaterialSpec

    with pytest.raises(ValueError, match="density"):
        MultilayerReflectivity(MaterialSpec("Ru"), 1.0, ("C", 2.52), 1.0, 10)


def test_compute_peak_fwhm_crossings_handles_missing_crossing() -> None:
    """A monotone rising curve has no bracketing half-max on one side."""

    x = np.linspace(0.0, 1.0, 11)
    y = np.linspace(0.0, 1.0, 11)
    fwhm, left, right, _ = mr._compute_peak_fwhm_crossings(x, y, peak_index=10)
    assert np.isnan(fwhm) and np.isnan(right)


def test_select_bragg_peak_index_prefers_local_max_right_of_bragg() -> None:
    """The strong local maximum above the Bragg angle wins over an earlier bump."""

    theta = np.linspace(0.5, 1.5, 101)
    rs2 = 0.2 * np.exp(-0.5 * ((theta - 0.7) / 0.02) ** 2)
    rs2 += 1.0 * np.exp(-0.5 * ((theta - 1.1) / 0.02) ** 2)
    mask = np.ones_like(theta, dtype=bool)
    idx = mr._select_bragg_peak_index(theta, rs2, mask, bragg_deg=1.0)
    assert idx is not None
    assert theta[idx] == pytest.approx(1.1, abs=1.0e-2)


def test_energy_from_bragg_angle_round_trips() -> None:
    """The angle-to-energy helper inverts the Bragg law for a known period."""

    period_angstrom = 27.0
    energy = mr._energy_from_bragg_angle(1.0, period_angstrom, order=1)
    wavelength_angstrom = mr.HC_EV_ANGSTROM / energy
    assert wavelength_angstrom == pytest.approx(2.0 * period_angstrom * np.sin(np.deg2rad(1.0)))
