"""File-based persistence for multilayer-optimization studies in the local web app.

A *study* is one directory under ``<data_dir>/multilayer_studies/``. Its three
stages write straight into it through the library's own layout
(``MultilayerOptimizationConfig(output_dir=<study dir>)`` derives
``0_d_spacing/``, ``1_gamma/``, ``2_blaze/``, ``plot/`` and
``optimization_state.json``). This module adds one ``study.json`` manifest on top
tracking the shared config and each stage's status, inputs and suggestions.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any

from grax.multilayer_optimization import MultilayerOptimizationConfig

from .persistence import _slugify

STAGES: tuple[str, ...] = ("d_spacing", "gamma", "blaze")
STAGE_LABELS: dict[str, str] = {
    "d_spacing": "1. D-spacing study",
    "gamma": "2. Gamma study",
    "blaze": "3. Blaze study",
}
STAGE_DIRNAMES: dict[str, str] = {
    "d_spacing": "0_d_spacing",
    "gamma": "1_gamma",
    "blaze": "2_blaze",
}
STAGE_PLOTS: dict[str, str] = {
    "d_spacing": "plot/0_d_spacing_study.png",
    "gamma": "plot/1_gamma_study.png",
    "blaze": "plot/2_blaze_study.png",
}
STAGE_CSVS: dict[str, str] = {
    "d_spacing": "0_d_spacing/d_spacing_study.csv",
    "gamma": "1_gamma/gamma_study.csv",
    "blaze": "2_blaze/blaze_study.csv",
}
# optimization_state.json keys each stage owns; cleared when a stage is reset.
STATE_KEYS_BY_STAGE: dict[str, tuple[str, ...]] = {
    "d_spacing": (
        "target_energy_eV",
        "wavelength_nm",
        "grating_grazing_angle_deg",
        "d_geometry_estimate_nm",
        "d_geometry_search_min_nm",
        "d_geometry_search_max_nm",
        "d_search_min_nm",
        "d_search_max_nm",
        "d_suggested_nm",
        "d_suggested_peak_rp",
        "d_reflectivity_best_nm",
        "d_reflectivity_best_peak_rp",
    ),
    "gamma": ("gamma_suggested", "gamma_suggested_peak_rp"),
    "blaze": ("blaze_suggested_deg", "blaze_suggested_efficiency"),
}
_TERMINAL_STAGE_STATES = {"completed", "aborted"}


def downstream_stages(stage: str) -> tuple[str, ...]:
    """Return the stages that run after ``stage``."""

    return STAGES[STAGES.index(stage) + 1 :]


@dataclass(frozen=True)
class FieldSpec:
    """One editable config field on the study forms.

    Attributes:
        name: ``MultilayerOptimizationConfig`` field name.
        kind: ``number`` / ``int`` / ``text`` / ``select`` / ``checkbox`` /
            ``material`` (a name + density pair).
        label: Human-readable label.
        section: Fieldset heading it belongs to.
        advanced: Rendered inside the collapsible "Advanced" section.
        choices: Options for ``select`` fields.
        stages: Which stage forms show this field (empty = the new-study form
            only).
    """

    name: str
    kind: str
    label: str
    section: str
    advanced: bool = False
    choices: tuple[str, ...] = ()
    stages: tuple[str, ...] = ()


STUDY_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("target_energy_ev", "number", "Target energy, eV", "Target & geometry"),
    FieldSpec("grating_density_lpermm", "number", "Line density, l/mm", "Target & geometry"),
    FieldSpec("diffraction_order", "int", "Diffraction order", "Target & geometry"),
    FieldSpec("cff", "number", "CFF", "Target & geometry"),
    FieldSpec("multilayer_bragg_order", "int", "Multilayer Bragg order", "Target & geometry"),
    FieldSpec("material_a", "material", "Material A (top)", "Materials"),
    FieldSpec("material_b", "material", "Material B", "Materials"),
    FieldSpec("substrate_material", "material", "Substrate", "Materials"),
    FieldSpec("n_bilayers", "int", "Bilayers", "Materials"),
    FieldSpec("solver", "select", "Solver", "Numerics", choices=("neviere", "rcwa")),
    FieldSpec("polarization", "select", "Polarization", "Numerics", choices=("p", "s")),
    FieldSpec("d_spacing_energy_min_ev", "number", "d-spacing energy min, eV", "Energy grids"),
    FieldSpec("d_spacing_energy_max_ev", "number", "d-spacing energy max, eV", "Energy grids"),
    FieldSpec("d_spacing_energy_step_ev", "number", "d-spacing energy step, eV", "Energy grids"),
    FieldSpec("gamma_energy_min_ev", "number", "gamma energy min, eV", "Energy grids"),
    FieldSpec("gamma_energy_max_ev", "number", "gamma energy max, eV", "Energy grids"),
    FieldSpec("gamma_energy_step_ev", "number", "gamma energy step, eV", "Energy grids"),
    FieldSpec("blaze_energy_min_ev", "number", "blaze energy min, eV", "Energy grids"),
    FieldSpec("blaze_energy_max_ev", "number", "blaze energy max, eV", "Energy grids"),
    FieldSpec("blaze_energy_points", "int", "blaze energy points", "Energy grids"),
    FieldSpec("bragg_angle_min_deg", "number", "Bragg angle min, deg", "Scan ranges"),
    FieldSpec("bragg_angle_max_deg", "number", "Bragg angle max, deg", "Scan ranges"),
    FieldSpec("d_spacing_relative_range", "number", "d relative range", "Scan ranges"),
    FieldSpec("d_spacing_min_practical_nm", "number", "d practical min, nm", "Scan ranges"),
    FieldSpec("d_spacing_max_practical_nm", "number", "d practical max, nm", "Scan ranges"),
    FieldSpec("d_spacing_points", "int", "d candidates", "Scan ranges"),
    FieldSpec("gamma_min", "number", "gamma min", "Scan ranges"),
    FieldSpec("gamma_max", "number", "gamma max", "Scan ranges"),
    FieldSpec("gamma_step", "number", "gamma step", "Scan ranges"),
    FieldSpec("blaze_angle_deg", "number", "Blaze center, deg", "Scan ranges"),
    FieldSpec("blaze_angle_half_range_deg", "number", "Blaze half-range, deg", "Scan ranges"),
    FieldSpec("blaze_angle_points", "int", "Blaze points", "Scan ranges"),
    FieldSpec("anti_blaze_angle_deg", "number", "Anti-blaze, deg (0 = sawtooth)", "Scan ranges"),
    FieldSpec("d_spacing_nm", "text", "d-spacing, nm (or 'auto')", "Selected values"),
    FieldSpec("gamma", "number", "gamma", "Selected values"),
    *(
        FieldSpec(name, kind, label, "Advanced", advanced=True)
        for name, kind, label in (
            ("rough_fourier_orders", "int", "Rough Fourier orders"),
            ("fine_fourier_orders", "int", "Fine Fourier orders"),
            ("final_fourier_orders", "int", "Final Fourier orders"),
            ("rough_scan_points", "int", "Rough scan points"),
            ("fine_scan_points", "int", "Fine scan points"),
            ("grax_x_resolution_nm", "number", "Grating x resolution, nm"),
            ("grax_z_resolution_nm", "number", "Grating z resolution, nm"),
            ("final_x_resolution_nm", "number", "Final x resolution, nm"),
            ("final_z_resolution_nm", "number", "Final z resolution, nm"),
            ("xrt_window_deg", "number", "XRT window, deg"),
            ("xrt_angle_points", "int", "XRT angle points"),
            ("roughness_sigma_nm", "number", "Roughness sigma, nm (blank = none)"),
        )
    ),
    FieldSpec("quick", "checkbox", "Quick mode (coarser grids)", "Advanced", advanced=True),
)

# Which fields each stage's inline form shows (the rest come from the study config).
_STAGE_FORM_FIELDS: dict[str, tuple[str, ...]] = {
    "d_spacing": (
        "target_energy_ev", "grating_density_lpermm", "diffraction_order", "cff",
        "multilayer_bragg_order", "material_a", "material_b", "substrate_material",
        "n_bilayers", "bragg_angle_min_deg", "bragg_angle_max_deg", "d_spacing_relative_range",
        "d_spacing_min_practical_nm", "d_spacing_max_practical_nm", "d_spacing_points",
        "gamma", "d_spacing_energy_min_ev", "d_spacing_energy_max_ev", "d_spacing_energy_step_ev",
    ),
    "gamma": (
        "d_spacing_nm", "gamma_min", "gamma_max", "gamma_step",
        "gamma_energy_min_ev", "gamma_energy_max_ev", "gamma_energy_step_ev",
        "solver", "polarization",
    ),
    "blaze": (
        "d_spacing_nm", "gamma", "blaze_angle_deg", "blaze_angle_half_range_deg",
        "blaze_angle_points", "anti_blaze_angle_deg", "blaze_energy_min_ev",
        "blaze_energy_max_ev", "blaze_energy_points", "solver", "polarization",
    ),
}


def _field_by_name() -> dict[str, FieldSpec]:
    return {spec.name: spec for spec in STUDY_FIELDS}


def stage_form_fields(stage: str) -> list[FieldSpec]:
    """Return the field specs shown on one stage's inline form."""

    lookup = _field_by_name()
    return [lookup[name] for name in _STAGE_FORM_FIELDS[stage]]


def study_form_sections(advanced: bool) -> list[tuple[str, list[FieldSpec]]]:
    """Return ``(section, fields)`` groups for the new-study form.

    Args:
        advanced: ``True`` for the advanced (collapsible) fields, ``False`` for
            the always-visible ones.

    Returns:
        Section groups preserving :data:`STUDY_FIELDS` order.
    """

    sections: dict[str, list[FieldSpec]] = {}
    for spec in STUDY_FIELDS:
        if bool(spec.advanced) != advanced:
            continue
        sections.setdefault(spec.section, []).append(spec)
    return list(sections.items())


def study_config_defaults() -> dict[str, Any]:
    """Return the JSON-safe config dict from ``MultilayerOptimizationConfig`` defaults."""

    dataclass_defaults = {f.name: f.default for f in fields(MultilayerOptimizationConfig)}
    config: dict[str, Any] = {}
    for spec in STUDY_FIELDS:
        default = dataclass_defaults.get(spec.name)
        if spec.kind == "material":
            name, density = default if isinstance(default, (tuple, list)) else ("", None)
            config[spec.name] = [str(name), float(density)]
        elif spec.kind == "checkbox":
            config[spec.name] = bool(default)
        elif spec.name == "roughness_sigma_nm":
            config[spec.name] = None if default is None else float(default)
        else:
            config[spec.name] = default
    return config


def _coerce_field(spec: FieldSpec, raw: str) -> Any:
    """Coerce one raw form value for ``spec`` into its JSON-safe type."""

    text = raw.strip()
    if spec.kind == "int":
        return int(float(text))
    if spec.kind == "number":
        if spec.name == "roughness_sigma_nm" and text == "":
            return None
        return float(text)
    if spec.name == "d_spacing_nm":
        return "auto" if text.lower() == "auto" else float(text)
    return text


def parse_study_config(form: Any, base: dict[str, Any] | None = None) -> dict[str, Any]:
    """Overlay a form's values onto ``base`` (or the defaults) and return a config dict."""

    config = dict(base or study_config_defaults())
    for spec in STUDY_FIELDS:
        if spec.kind == "material":
            name = form.get(f"{spec.name}_name")
            density = form.get(f"{spec.name}_density")
            if name is not None and density not in (None, ""):
                config[spec.name] = [str(name).strip(), float(density)]
            continue
        if spec.kind == "checkbox":
            if any(key == spec.name for key in form):
                config[spec.name] = form.get(spec.name) not in (None, "", "0", "false")
            elif base is None:
                config[spec.name] = False
            continue
        if spec.name in form and str(form.get(spec.name)).strip() != "":
            config[spec.name] = _coerce_field(spec, str(form.get(spec.name)))
    return config


def build_optimization_config(
    study_dir: Path, config: dict[str, Any]
) -> MultilayerOptimizationConfig:
    """Build a ``MultilayerOptimizationConfig`` for ``study_dir`` from a config dict."""

    kwargs: dict[str, Any] = {}
    for key, value in config.items():
        if key in {"material_a", "material_b", "substrate_material"} and isinstance(
            value, (list, tuple)
        ):
            kwargs[key] = (str(value[0]), float(value[1]))
        else:
            kwargs[key] = value
    return MultilayerOptimizationConfig(output_dir=study_dir, **kwargs)


class MultilayerStudyStore:
    """Store multilayer-optimization study manifests in a filesystem directory."""

    def __init__(self, directory: str | Path) -> None:
        """Initialise the store rooted at ``directory``."""

        self.directory = Path(directory)

    def list(self) -> list[dict[str, Any]]:
        """Return study manifests, newest first."""

        if not self.directory.exists():
            return []
        studies = [self.load(path.parent.name) for path in self.directory.glob("*/study.json")]
        return sorted(
            studies,
            key=lambda study: (str(study.get("created_at", "")), str(study.get("id", ""))),
            reverse=True,
        )

    def load(self, study_id: str) -> dict[str, Any]:
        """Load one study manifest by id."""

        path = self._study_dir(study_id) / "study.json"
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload.setdefault("id", study_id)
        return payload

    def save(self, manifest: dict[str, Any]) -> dict[str, Any]:
        """Persist a study manifest atomically and return it."""

        payload = dict(manifest)
        payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
        study_dir = self._study_dir(str(payload["id"]))
        study_dir.mkdir(parents=True, exist_ok=True)
        path = study_dir / "study.json"
        temp_path = path.with_name(f"study.json.{datetime.now().timestamp():.9f}.tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temp_path.replace(path)
        return payload

    def create(self, *, display_name: str, config: dict[str, Any]) -> dict[str, Any]:
        """Create a new study directory + manifest and return it."""

        slug = _slugify(display_name) or "study"
        study_id = f"{datetime.now():%Y%m%d-%H%M%S}-{slug}"
        candidate = study_id
        suffix = 2
        while (self._study_dir(candidate) / "study.json").exists():
            candidate = f"{study_id}-{suffix}"
            suffix += 1
        manifest = {
            "id": candidate,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "display_name": display_name.strip() or candidate,
            "comment": "",
            "config": config,
            "stages": {stage: _blank_stage() for stage in STAGES},
        }
        return self.save(manifest)

    def delete_many(self, study_ids: Sequence[str]) -> None:
        """Delete several study directories."""

        for study_id in study_ids:
            study_dir = self._study_dir(study_id)
            if study_dir.exists():
                shutil.rmtree(study_dir)

    def study_dir(self, study_id: str) -> Path:
        """Return the directory for one study id (validated)."""

        return self._study_dir(study_id)

    def _study_dir(self, study_id: str) -> Path:
        if _slugify(study_id) != study_id:
            raise ValueError("Invalid study id.")
        return self.directory / study_id


def _blank_stage() -> dict[str, Any]:
    return {
        "status": "not_run",
        "ran_at": None,
        "error_text": "",
        "aborted": False,
        "config_snapshot": None,
        "suggested": {},
        "artifacts": {},
    }
