"""Saved Bragg gratings: a crystal with comb electrodes, kept as JSON files."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .persistence import _slugify
from .files import remove_file, write_json_atomically

# The comb of the pulse-picking device of Vadilonga et al., Opt. Express 29,
# 34962 (2021): LiNbO3 with 1 um fingers and gaps on a (104) surface.
DEFAULTS: dict[str, Any] = {
    "name": "",
    "label": "",
    "crystal": "LiNbO3",
    "h": 1,
    "k": 0,
    "l": 4,
    "finger_um": 1.0,
    "gap_um": 1.0,
    "al_thickness_um": 0.1,
    "penetration_um": 1.59,
    "electrode_length_um": 1200.0,
    "aperture_um": 200.0,
    "coated": False,
    "coating_blocks": [],
}

# Shown when the coating is first switched on: a Pt mirror coating.
DEFAULT_COATING_BLOCKS: tuple[dict[str, str], ...] = (
    {"kind": "single", "material": "Pt", "density_g_cm3": "", "thickness_nm": "30", "planarization_nm": "0"},
)

_BLOCK_FIELDS = (
    "kind", "material", "density_g_cm3", "thickness_nm", "planarization_nm",
    "material_b", "density_b_g_cm3", "thickness_b_nm", "repeats",
)
_MAX_COATING_LAYERS = 1000


def crystal_choices() -> tuple[str, ...]:
    """Crystals the model knows."""

    from ..bragg_grating import CRYSTALS

    return CRYSTALS


def _number(form: Any, name: str, low: float, high: float, label: str) -> float:
    raw = str(form.get(name, DEFAULTS[name])).strip().replace(",", ".")
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{label} must be a number.") from None
    if not np.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{label} must be between {low:g} and {high:g}.")
    return value


def geometry_from_form(form: Any) -> dict[str, float]:
    """The electrode geometry of a grating form; raise ``ValueError`` with a message for the user."""

    return {
        "finger_um": _number(form, "finger_um", 0.05, 50.0, "Finger width"),
        "gap_um": _number(form, "gap_um", 0.05, 50.0, "Gap width"),
        "al_thickness_um": _number(form, "al_thickness_um", 0.0, 2.0, "Al thickness"),
        "penetration_um": _number(form, "penetration_um", 0.05, 50.0, "Penetration depth"),
        "electrode_length_um": _number(form, "electrode_length_um", 1.0, 100000.0, "Electrode length"),
        "aperture_um": _number(form, "aperture_um", 1.0, 100000.0, "Electrode aperture"),
    }


def spec_from_form(form: Any) -> dict[str, Any]:
    """Read and validate a grating form; raise ``ValueError`` with a message for the user."""

    def index(name: str, label: str) -> int:
        try:
            value = int(str(form.get(name, DEFAULTS[name])).strip())
        except ValueError:
            raise ValueError(f"{label} must be a whole number.") from None
        if abs(value) > 20:
            raise ValueError(f"{label} must be between -20 and 20.")
        return value

    name = str(form.get("name", "")).strip()
    if not name:
        raise ValueError("Give the grating a name.")
    if len(name) > 120:
        raise ValueError("The name must be at most 120 characters.")
    crystal = str(form.get("crystal", DEFAULTS["crystal"])).strip()
    if crystal not in crystal_choices():
        raise ValueError(f"Choose a crystal from {', '.join(crystal_choices())}.")
    hkl = (index("h", "h"), index("k", "k"), index("l", "l"))
    _require_allowed_reflection(crystal, hkl)
    return {
        "name": name,
        "label": str(form.get("label", "")).strip()[:300],
        "crystal": crystal,
        "h": hkl[0],
        "k": hkl[1],
        "l": hkl[2],
        **geometry_from_form(form),
        **coating_from_form(form),
    }


def coating_from_form(form: Any) -> dict[str, Any]:
    """The coating part of a grating form, as saved: on/off and the blocks top to bottom.

    Raises ``ValueError`` with a message for the user when a block is invalid.
    """

    coated = str(form.get("coated", "")).strip().lower() in {"1", "on", "true", "yes"}
    blocks = coating_from_form_rows(form) if hasattr(form, "getlist") else []
    if coated:
        if not blocks:
            raise ValueError("Add at least one coating layer, or switch the coating off.")
        coating_layers({"coated": True, "coating_blocks": blocks})
    return {"coated": coated, "coating_blocks": blocks}


def coating_layers(spec: dict[str, Any]) -> tuple[Any, ...]:
    """Coating layers of a saved grating from the bottom up (empty when uncoated)."""

    from ..idt_surface_grating import CoatingLayer
    from .plane_mirror import _material_from_text

    if not spec.get("coated"):
        return ()

    def number(text: str, label: str, *, minimum: float, allow_equal: bool) -> float:
        try:
            value = float(str(text).replace(",", "."))
        except ValueError:
            raise ValueError(f"{label} must be a number.") from None
        if not np.isfinite(value) or value < minimum or (value == minimum and not allow_equal):
            relation = "at least" if allow_equal else "above"
            raise ValueError(f"{label} must be {relation} {minimum:g}.")
        return value

    def layer(material: str, density: str, thickness: str, planarization: float, label: str) -> Any:
        return CoatingLayer(
            _material_from_text(material, density, label),
            number(thickness, f"{label}: thickness", minimum=0.0, allow_equal=False),
            planarization,
        )

    top_down: list[Any] = []
    for index, block in enumerate(spec.get("coating_blocks", [])):
        label = f"Coating block {index + 1}"
        planarization = number(
            block.get("planarization_nm") or "0", f"{label}: planarization", minimum=0.0, allow_equal=True
        )
        first = layer(block.get("material", ""), block.get("density_g_cm3", ""), block.get("thickness_nm", ""), planarization, label + " A")
        if block.get("kind", "single") == "single":
            top_down.append(first)
            continue
        if block.get("kind") != "multilayer":
            raise ValueError(f"{label}: choose Single layer or Multilayer.")
        try:
            repeats = int(block.get("repeats", "1"))
        except ValueError:
            raise ValueError(f"{label}: bilayer repeats must be a whole number.") from None
        if not 1 <= repeats <= 250:
            raise ValueError(f"{label}: bilayer repeats must be between 1 and 250.")
        second = layer(block.get("material_b", ""), block.get("density_b_g_cm3", ""), block.get("thickness_b_nm", ""), planarization, label + " B")
        top_down.extend([first, second] * repeats)
        if len(top_down) > _MAX_COATING_LAYERS:
            raise ValueError(f"The coating may have at most {_MAX_COATING_LAYERS} layers.")
    return tuple(top_down[::-1])


def coating_rows(values: Any) -> list[dict[str, str]]:
    """Blocks to fill the coating editor with, top to bottom."""

    if hasattr(values, "getlist"):
        blocks = coating_from_form_rows(values)
    else:
        blocks = values.get("coating_blocks")
    return [dict(block) for block in (blocks or DEFAULT_COATING_BLOCKS)]


def coating_from_form_rows(form: Any) -> list[dict[str, str]]:
    """The posted coating blocks without validation, to refill a form after an error."""

    count = len(form.getlist("cl_material"))
    rows = []
    for index in range(count):
        row = {}
        for name in _BLOCK_FIELDS:
            values = form.getlist("cl_" + name)
            default = "single" if name == "kind" else "1" if name == "repeats" else ""
            row[name] = str(values[index]).strip() if index < len(values) else default
        rows.append(row)
    return rows


def surface_grating_from_spec(spec: dict[str, Any], amplitude_nm: float = 0.0, **options: Any) -> Any:
    """The coated device of a saved grating for the grazing-incidence solvers."""

    from ..idt_surface_grating import CRYSTAL_MATERIALS, IdtSurfaceGrating

    return IdtSurfaceGrating(
        idt=idt_grating_from_spec(spec),
        amplitude_nm=float(amplitude_nm),
        crystal_material=CRYSTAL_MATERIALS[spec["crystal"]],
        layers=coating_layers(spec),
        **options,
    )


def has_multilayer(spec: dict[str, Any]) -> bool:
    """Whether the coating holds a repeated pair of layers with a Bragg peak."""

    from ..idt_surface_grating import multilayer_period_nm

    if not spec.get("coated"):
        return False
    try:
        return multilayer_period_nm(coating_layers(spec)) is not None
    except ValueError:
        return False


def default_solver(spec: dict[str, Any]) -> str:
    """Takagi-Taupin for a bare crystal, RCWA once there is a coating."""

    return "rcwa" if spec.get("coated") and spec.get("coating_blocks") else "takagi"


def coating_summary(spec: dict[str, Any]) -> str:
    """Short description of the coating, or an empty string."""

    if not spec.get("coated"):
        return ""
    parts = []
    for block in spec.get("coating_blocks", []):
        if block.get("kind") == "multilayer":
            parts.append(
                f"[{block.get('material')} {block.get('thickness_nm')} / {block.get('material_b')} "
                f"{block.get('thickness_b_nm')} nm] × {block.get('repeats')}"
            )
        else:
            parts.append(f"{block.get('material')} {block.get('thickness_nm')} nm")
    return "coating " + " on ".join(parts) if parts else ""


def _require_allowed_reflection(crystal: str, hkl: tuple[int, int, int]) -> None:
    """Refuse a reflection the crystal symmetry forbids (reachability depends on the energy)."""

    from ..bragg_grating import bragg_crystal

    try:
        bragg_crystal(crystal, hkl, 30000.0)
    except ValueError as error:
        if "cannot be reached" not in str(error):
            raise


def idt_grating_from_spec(spec: dict[str, Any]) -> Any:
    """The model object of a saved grating."""

    from ..bragg_grating import IdtGrating

    return IdtGrating(
        finger_um=float(spec["finger_um"]),
        gap_um=float(spec["gap_um"]),
        penetration_um=float(spec["penetration_um"]),
        al_thickness_um=float(spec["al_thickness_um"]),
        electrode_length_um=float(spec["electrode_length_um"]),
        aperture_um=float(spec["aperture_um"]),
    )


def grating_summary(spec: dict[str, Any]) -> str:
    """One line describing a saved grating."""

    period = 2.0 * (float(spec["finger_um"]) + float(spec["gap_um"]))
    return (
        f"{spec['crystal']} ({spec['h']} {spec['k']} {spec['l']}) · "
        f"period {period:g} µm · fingers {spec['finger_um']:g} µm, gaps {spec['gap_um']:g} µm"
        + (f" · {coating_summary(spec)}" if coating_summary(spec) else "")
    )


class BraggGratingStore:
    """Saved Bragg gratings as one JSON file each."""

    def __init__(self, directory: str | Path) -> None:
        """Keep the gratings in ``directory``."""

        self.directory = Path(directory)

    def list(self) -> list[dict[str, Any]]:
        """Saved gratings ordered by name."""

        if not self.directory.exists():
            return []
        specs = []
        for path in self.directory.glob("*.json"):
            try:
                specs.append(self.load(path.stem))
            except (OSError, ValueError):
                continue
        return sorted(specs, key=lambda spec: str(spec.get("name", "")).lower())

    def load(self, grating_id: str) -> dict[str, Any]:
        """One grating by id; raises ``FileNotFoundError`` when unknown."""

        with self._path(grating_id).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def save(self, spec: dict[str, Any], grating_id: str | None = None) -> dict[str, Any]:
        """Create a grating, or replace the one with ``grating_id``; return what was saved."""

        now = datetime.now().isoformat(timespec="seconds")
        payload = dict(spec)
        if grating_id is None:
            payload["id"] = self._unique_id(str(spec["name"]))
            payload["created_at"] = now
        else:
            previous = self.load(grating_id)
            payload["id"] = grating_id
            payload["created_at"] = previous.get("created_at", now)
        payload["updated_at"] = now
        self.directory.mkdir(parents=True, exist_ok=True)
        write_json_atomically(self._path(payload["id"]), payload, sort_keys=True)
        return payload

    def delete(self, grating_id: str) -> None:
        """Remove a grating if it exists."""

        remove_file(self._path(grating_id))

    def _path(self, grating_id: str) -> Path:
        if _slugify(grating_id) != grating_id:
            raise ValueError("Invalid grating id.")
        return self.directory / f"{grating_id}.json"

    def _unique_id(self, name: str) -> str:
        base = _slugify(name) or "grating"
        candidate, number = base, 2
        while (self.directory / f"{candidate}.json").exists():
            candidate = f"{base}-{number}"
            number += 1
        return candidate
