"""Persistence and form helpers for web multilayer-design studies.

One *study* is a directory under ``<data_dir>/multilayer_designs/<study_id>/``.
:class:`grax.MultilayerDesignConfig`'s ``output_dir`` points straight at that
directory, so the library's own layout (``survey/``, ``plots/``,
``energy_scan/``) lands inside it. This module adds exactly one extra file,
``study.json``, holding the config the user entered plus per-stage status.

The workflow has two stages, matching the library:

* ``survey`` -- :meth:`grax.MultilayerGratingDesigner.run_survey`, one
  single-energy theta search per ``(d, blaze)`` grid cell.
* ``energy_scan`` -- :meth:`grax.MultilayerGratingDesigner.run_energy_scan`
  over the ``(d, blaze)`` designs picked off the survey.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any

from grax.multilayer_design import MultilayerDesignConfig, ThetaSearchScanSettings

from .persistence import _slugify

STAGES: tuple[str, ...] = ("survey", "energy_scan")

STAGE_LABELS: dict[str, str] = {
    "survey": "1. d-spacing x blaze-angle survey",
    "energy_scan": "2. Energy scan of the chosen designs",
}

#: Selection modes the energy-scan stage accepts.
DESIGN_MODES: tuple[str, ...] = ("best", "per_d", "manual")

_TERMINAL_STAGE_STATES = {"completed", "aborted"}

#: Survey cells above this count get a warning on the creation form.
SURVEY_CELL_WARNING_THRESHOLD = 200


def downstream_stages(stage: str) -> tuple[str, ...]:
    """Return the stages that depend on ``stage``."""

    return STAGES[STAGES.index(stage) + 1 :]


# --------------------------------------------------------------------------- #
# Form field specs                                                            #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FieldSpec:
    """One editable config field on the study form.

    Attributes:
        name: :class:`grax.MultilayerDesignConfig` field name. A dotted name
            such as ``"survey_scan_settings.rough_scan_points"`` addresses a
            field on one of the nested :class:`grax.ThetaSearchScanSettings`.
        kind: ``number`` / ``int`` / ``text`` / ``select`` / ``checkbox`` /
            ``material`` (a name + density pair).
        label: Human-readable label.
        section: Fieldset heading it belongs to.
        advanced: Rendered inside the collapsible "Advanced" section.
        choices: Options for ``select`` fields.
        row: Optional sub-group inside ``section``. Fields sharing a ``row``
            render on one line under that heading; ``""`` means "no sub-group",
            and those fields flow in the fieldset as usual.
    """

    name: str
    kind: str
    label: str
    section: str
    advanced: bool = False
    choices: tuple[str, ...] = ()
    row: str = ""

    @property
    def html_name(self) -> str:
        """Return the form input name (dots are legal in HTML field names)."""

        return self.name


def _scan_fields(prefix: str, section: str) -> tuple[FieldSpec, ...]:
    """Return the advanced field specs for one nested scan-settings block.

    Grouped into one line per theta-search pass -- rough, then fine, then the
    final solve -- followed by the two settings that apply to the search as a
    whole. The pass name lives in the row heading, so the field labels drop it.
    """

    return (
        FieldSpec(f"{prefix}.rough_scan_half_width_deg", "number",
                  "Half-width, deg", section, advanced=True, row="Rough pass"),
        FieldSpec(f"{prefix}.rough_scan_points", "int",
                  "Points", section, advanced=True, row="Rough pass"),
        FieldSpec(f"{prefix}.rough_fourier_orders", "int",
                  "Fourier orders", section, advanced=True, row="Rough pass"),
        FieldSpec(f"{prefix}.rough_x_resolution_nm", "number",
                  "x resolution, nm", section, advanced=True, row="Rough pass"),
        FieldSpec(f"{prefix}.rough_z_resolution_nm", "number",
                  "z resolution, nm", section, advanced=True, row="Rough pass"),
        FieldSpec(f"{prefix}.fine_scan_half_width_deg", "number",
                  "Half-width, deg", section, advanced=True, row="Fine pass"),
        FieldSpec(f"{prefix}.fine_scan_points", "int",
                  "Points", section, advanced=True, row="Fine pass"),
        FieldSpec(f"{prefix}.fine_fourier_orders", "int",
                  "Fourier orders", section, advanced=True, row="Fine pass"),
        FieldSpec(f"{prefix}.fine_x_resolution_nm", "number",
                  "x resolution, nm", section, advanced=True, row="Fine pass"),
        FieldSpec(f"{prefix}.fine_z_resolution_nm", "number",
                  "z resolution, nm", section, advanced=True, row="Fine pass"),
        FieldSpec(f"{prefix}.final_fourier_orders", "int",
                  "Fourier orders", section, advanced=True, row="Final solve"),
        FieldSpec(f"{prefix}.final_x_resolution_nm", "number",
                  "x resolution, nm", section, advanced=True, row="Final solve"),
        FieldSpec(f"{prefix}.final_z_resolution_nm", "number",
                  "z resolution, nm", section, advanced=True, row="Final solve"),
        FieldSpec(f"{prefix}.precise_peak_selection_mode", "select",
                  "Peak selection", section, advanced=True, row="Peak & roughness",
                  choices=("max", "gauss", "voigt")),
        FieldSpec(f"{prefix}.roughness_sigma_nm", "number",
                  "Roughness sigma, nm (blank = none)", section, advanced=True,
                  row="Peak & roughness"),
    )


#: Every editable field, in render order. Sections mirror the three groups in
#: ``MultilayerDesignConfig``: shared, survey-only, energy-scan-only.
STUDY_FIELDS: tuple[FieldSpec, ...] = (
    # -- Shared --------------------------------------------------------- #
    FieldSpec("grating_density_lpermm", "number", "Line density, l/mm",
              "Shared - geometry & order"),
    FieldSpec("diffraction_order", "int", "Diffraction order",
              "Shared - geometry & order"),
    FieldSpec("multilayer_bragg_order", "int", "Multilayer Bragg order",
              "Shared - geometry & order"),
    FieldSpec("anti_blaze_angle_deg", "number", "Anti-blaze angle, deg (0 = sawtooth)",
              "Shared - geometry & order"),
    FieldSpec("material_a", "material", "Material A (top)", "Shared - materials"),
    FieldSpec("material_b", "material", "Material B", "Shared - materials"),
    FieldSpec("substrate_material", "material", "Substrate", "Shared - materials"),
    FieldSpec("n_bilayers", "int", "Bilayers", "Shared - materials"),
    FieldSpec("gamma", "number", "Gamma (material A fraction)", "Shared - materials"),
    FieldSpec("coating_label", "text", "Coating label for plots (e.g. Ru/B4C)",
              "Shared - materials"),
    FieldSpec("solver", "select", "Solver", "Shared - numerics",
              choices=("neviere", "rcwa")),
    FieldSpec("polarization", "select", "Polarization", "Shared - numerics",
              choices=("p", "s")),
    FieldSpec("backend", "select", "Backend", "Shared - numerics",
              choices=("numba", "numpy")),
    FieldSpec("x_resolution_nm", "number", "Grating x resolution, nm",
              "Shared - numerics"),
    FieldSpec("z_resolution_nm", "number", "Grating z resolution, nm",
              "Shared - numerics"),
    FieldSpec("checkpoint", "checkbox", "Write checkpoints", "Shared - runtime",
              advanced=True),
    FieldSpec("resume", "checkbox", "Resume from checkpoints", "Shared - runtime",
              advanced=True),
    FieldSpec("save_profile_plot", "checkbox", "Save profile plots", "Shared - runtime",
              advanced=True),
    FieldSpec("save_stack_plot", "checkbox", "Save stack plots", "Shared - runtime",
              advanced=True),
    # -- Survey only ---------------------------------------------------- #
    FieldSpec("target_energy_ev", "number", "Survey energy, eV",
              "Survey - target & grids"),
    FieldSpec("d_min_nm", "number", "d-spacing min, nm", "Survey - target & grids"),
    FieldSpec("d_max_nm", "number", "d-spacing max, nm", "Survey - target & grids"),
    FieldSpec("d_points", "int", "d-spacing points", "Survey - target & grids"),
    FieldSpec("blaze_min_deg", "number", "Blaze min, deg", "Survey - target & grids"),
    FieldSpec("blaze_max_deg", "number", "Blaze max, deg", "Survey - target & grids"),
    FieldSpec("blaze_points", "int", "Blaze points", "Survey - target & grids"),
    FieldSpec("on_error", "select", "On a failing cell", "Survey - target & grids",
              choices=("continue", "fail_fast")),
    *_scan_fields("survey_scan_settings", "Survey - theta-search settings"),
    # -- Energy scan only ----------------------------------------------- #
    FieldSpec("energy_scan_min_ev", "number", "Energy min, eV",
              "Energy scan - energy grid"),
    FieldSpec("energy_scan_max_ev", "number", "Energy max, eV",
              "Energy scan - energy grid"),
    FieldSpec("energy_scan_points", "int", "Energy points",
              "Energy scan - energy grid"),
    FieldSpec("max_workers", "text", "Max workers (\"auto\" or a number)",
              "Energy scan - runtime", advanced=True),
    FieldSpec("theta_tracking_mode", "select", "Theta tracking",
              "Energy scan - runtime", advanced=True,
              choices=("auto", "bragg", "previous")),
    FieldSpec("max_tracking_energy_step_ev", "number",
              "Max tracking energy step, eV (blank = none)",
              "Energy scan - runtime", advanced=True),
    *_scan_fields("energy_scan_settings", "Energy scan - theta-search settings"),
)

#: Fields whose blank form value means ``None`` rather than "leave unchanged".
_NULLABLE_FIELDS = frozenset(
    {
        "coating_label",
        "max_tracking_energy_step_ev",
        "survey_scan_settings.roughness_sigma_nm",
        "energy_scan_settings.roughness_sigma_nm",
    }
)

_MATERIAL_FIELDS = frozenset({"material_a", "material_b", "substrate_material"})


def _field_by_name() -> dict[str, FieldSpec]:
    return {spec.name: spec for spec in STUDY_FIELDS}


def study_form_sections(advanced: bool) -> list[tuple[str, list[tuple[str, list[FieldSpec]]]]]:
    """Return ``(section, rows)`` groups for the study form.

    Each section holds ``(row_label, fields)`` pairs. A row labelled ``""``
    is an ungrouped run of fields that flows in the fieldset as usual; a named
    row renders on its own line under that heading (see
    :attr:`FieldSpec.row`).

    Args:
        advanced: ``True`` for the advanced (collapsible) fields, ``False`` for
            the always-visible ones.

    Returns:
        Section groups preserving :data:`STUDY_FIELDS` order.
    """

    sections: dict[str, list[tuple[str, list[FieldSpec]]]] = {}
    for spec in STUDY_FIELDS:
        if bool(spec.advanced) != advanced:
            continue
        rows = sections.setdefault(spec.section, [])
        if rows and rows[-1][0] == spec.row:
            rows[-1][1].append(spec)
        else:
            rows.append((spec.row, [spec]))
    return list(sections.items())


def _nested_defaults() -> dict[str, Any]:
    """Return the default values of both nested scan-settings blocks."""

    return {
        name: spec.default
        for name, spec in ThetaSearchScanSettings.__dataclass_fields__.items()
    }


def study_config_defaults() -> dict[str, Any]:
    """Return the JSON-safe config dict from :class:`grax.MultilayerDesignConfig` defaults."""

    dataclass_defaults = {f.name: f.default for f in fields(MultilayerDesignConfig)}
    scan_defaults = _nested_defaults()
    config: dict[str, Any] = {
        "survey_scan_settings": dict(scan_defaults),
        "energy_scan_settings": dict(scan_defaults),
    }
    for spec in STUDY_FIELDS:
        if "." in spec.name:
            continue
        default = dataclass_defaults.get(spec.name)
        if spec.kind == "material":
            name, density = default if isinstance(default, (tuple, list)) else ("", None)
            config[spec.name] = [str(name), float(density)]
        elif spec.kind == "checkbox":
            config[spec.name] = bool(default)
        else:
            config[spec.name] = default
    return config


def _coerce_field(spec: FieldSpec, raw: str) -> Any:
    """Coerce one raw form value for ``spec`` into its JSON-safe type."""

    text = raw.strip()
    if text == "" and spec.name in _NULLABLE_FIELDS:
        return None
    if spec.name == "max_workers":
        return text if text.lower() == "auto" else int(float(text))
    if spec.kind == "int":
        return int(float(text))
    if spec.kind == "number":
        return float(text)
    return text


def _set_nested(config: dict[str, Any], dotted: str, value: Any) -> None:
    """Assign ``value`` at a ``a.b`` path inside ``config``."""

    head, _, tail = dotted.partition(".")
    if not tail:
        config[dotted] = value
        return
    block = dict(config.get(head) or {})
    block[tail] = value
    config[head] = block


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
            # The form renders a hidden "0" before the box, so a checked box
            # submits both and the last value is the real one.
            values = form.getlist(spec.name) if hasattr(form, "getlist") else None
            if values:
                config[spec.name] = str(values[-1]) not in ("", "0", "false")
            elif any(key == spec.name for key in form):
                config[spec.name] = form.get(spec.name) not in (None, "", "0", "false")
            elif base is None:
                config[spec.name] = False
            continue
        if spec.name not in form:
            continue
        raw = str(form.get(spec.name))
        if raw.strip() == "" and spec.name not in _NULLABLE_FIELDS:
            continue
        _set_nested(config, spec.name, _coerce_field(spec, raw))
    return config


def build_design_config(study_dir: Path, config: dict[str, Any]) -> MultilayerDesignConfig:
    """Build a :class:`grax.MultilayerDesignConfig` for ``study_dir`` from a config dict."""

    kwargs: dict[str, Any] = {}
    for key, value in config.items():
        if key in _MATERIAL_FIELDS and isinstance(value, (list, tuple)):
            kwargs[key] = (str(value[0]), float(value[1]))
        elif key in {"survey_scan_settings", "energy_scan_settings"}:
            kwargs[key] = ThetaSearchScanSettings(**dict(value or {}))
        else:
            kwargs[key] = value
    return MultilayerDesignConfig(output_dir=study_dir, **kwargs)


def flatten_config_values(config: dict[str, Any]) -> dict[str, Any]:
    """Return ``config`` keyed by :class:`FieldSpec` names, dotted paths included.

    The templates render one input per :data:`STUDY_FIELDS` entry, so they need
    a flat lookup in which ``"survey_scan_settings.rough_scan_points"`` resolves
    without walking into the nested block.
    """

    flat: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, dict):
            for inner_key, inner_value in value.items():
                flat[f"{key}.{inner_key}"] = inner_value
        else:
            flat[key] = value
    return flat


#: Fields that change nothing a stage computed -- only plot labelling, which
#: artifacts get written, or how the work is scheduled. Editing one of these
#: leaves existing results valid.
_COSMETIC_FIELDS = frozenset(
    {
        "coating_label",
        "save_profile_plot",
        "save_stack_plot",
        "checkpoint",
        "resume",
        "max_workers",
    }
)


def stages_invalidated_by(old: dict[str, Any], new: dict[str, Any]) -> tuple[str, ...]:
    """Return the stages whose stored results no longer match an edited config.

    A field's :attr:`FieldSpec.section` says which stage reads it: the shared
    section feeds both, so changing one invalidates the survey and everything
    downstream of it.

    Args:
        old: The config as stored before the edit.
        new: The config after the edit.

    Returns:
        Stage names in :data:`STAGES` order, empty when nothing that affects a
        result changed.
    """

    before = flatten_config_values(old)
    after = flatten_config_values(new)
    invalidated: set[str] = set()
    for spec in STUDY_FIELDS:
        if spec.name in _COSMETIC_FIELDS or before.get(spec.name) == after.get(spec.name):
            continue
        if spec.section.startswith("Energy scan"):
            invalidated.add("energy_scan")
        else:
            invalidated.update(("survey", *downstream_stages("survey")))
    return tuple(stage for stage in STAGES if stage in invalidated)


def survey_cell_count(config: dict[str, Any]) -> int:
    """Return how many theta searches the survey grid implies."""

    try:
        return int(config.get("d_points", 0)) * int(config.get("blaze_points", 0))
    except (TypeError, ValueError):
        return 0



# --------------------------------------------------------------------------- #
# Offline script generation                                                   #
# --------------------------------------------------------------------------- #
#: Section prefix -> (banner title, banner note) for the generated script.
_SCRIPT_SECTIONS: tuple[tuple[str, str, str], ...] = (
    (
        "Shared",
        "SHARED -- grating geometry, materials, solver/runtime.",
        "Read by both stages.",
    ),
    (
        "Survey",
        "SURVEY -- read only by --survey (run_survey / evaluate_survey).",
        "",
    ),
    (
        "Energy scan",
        "ENERGY SCAN -- read only by --energy-scan.",
        "",
    ),
)

#: Nested scan-settings blocks, as ``config key -> generated constant``.
_SCRIPT_SCAN_BLOCKS = {
    "survey_scan_settings": "SURVEY_SCAN",
    "energy_scan_settings": "ENERGY_SCAN_SCAN",
}


def _script_literal(name: str, value: Any) -> str:
    """Return ``value`` as Python source, as a tuple for the material pairs."""

    if name in _MATERIAL_FIELDS and isinstance(value, (list, tuple)):
        return f"({str(value[0])!r}, {float(value[1])!r})"
    return repr(value)


def _script_banner(title: str, note: str) -> str:
    """Return a boxed section banner in the examples' style."""

    width = 75
    lines = [f"# {'=' * width} #", f"# {title.ljust(width)} #"]
    if note:
        lines.append(f"# {note.ljust(width)} #")
    lines.append(f"# {'=' * width} #")
    return "\n".join(lines)


_OFFLINE_SCRIPT_TEMPLATE = '''"""{display_name} -- standalone multilayer-grating design run.

Generated by the grax web app from study {study_id} on {generated_at}.
Everything this needs is in this one file: edit the parameters below, then run
the stages you want. It imports nothing but ``grax`` and pandas, so it can be
copied to another machine and run there.

Stage 1 (``--survey``) scans a 2-D grid of bilayer d-spacing x blaze angle. For
every pair it builds the multilayer-coated blazed grating and runs a
single-energy multilayer theta search at TARGET_ENERGY_EV -- the search scans
the incident angle and returns the one that maximizes the selected-order
efficiency, so there is no CFF input. It writes ``results/survey/survey.csv``,
three headline plots under ``results/plots/`` and a per-run folder tree under
``results/survey/runs/``.

Stage 2 (``--energy-scan``) sweeps ``(d, blaze)`` designs over
[ENERGY_SCAN_MIN_EV, ENERGY_SCAN_MAX_EV]: the optimal blaze per d-spacing by
default, only the best cell with ``--best``, or exactly ``--pairs``. Two or
more designs also get an overlay comparison plot.

With no stage flag both stages run in order. ``--eval`` re-derives results from
what is already on disk without solving anything.

Examples::

    python {script_name} --survey
    python {script_name} --energy-scan --best
    python {script_name} --energy-scan --pairs "3.0,0.8; 3.3,0.9"
    python {script_name}                      # survey, then every optimal blaze
    python {script_name} --survey --eval      # rebuild plots from existing runs

The executable body is guarded because the theta-search sweep spawns worker
processes that re-import this file by path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from grax import (
    MultilayerDesignConfig,
    MultilayerGratingDesigner,
    ThetaSearchScanSettings,
)

{parameters}

CONFIG = MultilayerDesignConfig(
{constructor}
)


def _parse_pairs(text: str) -> list[tuple[float, float]]:
    """Parse ``"d,blaze; d,blaze"`` into ``(d, blaze)`` float pairs."""

    pairs: list[tuple[float, float]] = []
    for chunk in text.replace("\\\\n", ";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        d_text, blaze_text = chunk.split(",")
        pairs.append((float(d_text), float(blaze_text)))
    if not pairs:
        raise ValueError(f"no (d, blaze) pairs parsed from {{text!r}}")
    return pairs


def _survey_table() -> pd.DataFrame:
    """Return the usable rows of ``results/survey/survey.csv``."""

    csv_path = CONFIG.survey_dir / "survey.csv"
    if not csv_path.is_file():
        raise SystemExit(f"{{csv_path}} not found -- run --survey first, or pass --pairs.")
    table = pd.read_csv(csv_path).dropna(subset=["peak_efficiency"])
    if table.empty:
        raise SystemExit("survey.csv has no usable rows; pass --pairs explicitly.")
    return table


def _optimal_pairs_from_survey() -> list[tuple[float, float]]:
    """Return one ``(d, optimal_blaze)`` pair per d-spacing."""

    table = _survey_table()
    best = table.loc[table.groupby("d_nm")["peak_efficiency"].idxmax()]
    return [(float(row.d_nm), float(row.blaze_deg)) for row in best.itertuples()]


def _best_pair_from_survey() -> list[tuple[float, float]]:
    """Return only the single ``(d, blaze)`` with the highest survey efficiency."""

    row = _survey_table().loc[lambda frame: frame["peak_efficiency"].idxmax()]
    print(
        f"Best survey design: d = {{float(row.d_nm):.3f}} nm, "
        f"blaze = {{float(row.blaze_deg):.3f}} deg "
        f"(efficiency {{float(row.peak_efficiency):.4g}})"
    )
    return [(float(row.d_nm), float(row.blaze_deg))]


def run_survey(designer: MultilayerGratingDesigner, *, evaluate: bool) -> None:
    """Run stage 1, or rebuild it from the runs already on disk."""

    result = designer.evaluate_survey() if evaluate else designer.run_survey()
    print(f"Survey table: {{result.combined_csv_path}}")
    print(f"  optimal blaze vs d: {{result.plot_path}}")
    print(f"  max efficiency vs d: {{result.efficiency_plot_path}}")
    print(f"  (d, blaze) heatmap:  {{result.heatmap_plot_path}}")


def run_energy_scan(
    designer: MultilayerGratingDesigner,
    *,
    evaluate: bool,
    pairs_text: str | None,
    best_only: bool,
) -> None:
    """Run stage 2 for the selected designs, or re-read existing results."""

    if evaluate and not (pairs_text or best_only):
        results = designer.evaluate_energy_scan()
    else:
        if pairs_text:
            pairs = _parse_pairs(pairs_text)
        elif best_only:
            pairs = _best_pair_from_survey()
        else:
            pairs = _optimal_pairs_from_survey()
        results = (
            designer.evaluate_energy_scan(pairs)
            if evaluate
            else designer.run_energy_scan(pairs)
        )

    for scan in results:
        print(
            f"d = {{scan.d_spacing_nm:.3f}} nm, blaze = {{scan.blaze_angle_deg:.3f}} deg "
            f"-> {{scan.summary_csv_path}}"
        )
        print(f"  plot: {{scan.titled_plot_path}}")


def main() -> None:
    """Parse the stage flags and run what was asked for."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey", action="store_true", help="Run stage 1.")
    parser.add_argument(
        "--energy-scan", action="store_true", dest="energy_scan", help="Run stage 2."
    )
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument(
        "--pairs", default=None, help='Semicolon-separated "d_nm,blaze_deg" designs.'
    )
    selector.add_argument(
        "--best", action="store_true", help="Scan only the best surveyed design."
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        dest="evaluate",
        help="Re-derive results from disk instead of solving.",
    )
    args = parser.parse_args()

    # No stage flag means the whole workflow, in order.
    survey = args.survey or not (args.survey or args.energy_scan)
    energy_scan = args.energy_scan or not (args.survey or args.energy_scan)

    designer = MultilayerGratingDesigner(CONFIG)
    if survey:
        run_survey(designer, evaluate=args.evaluate)
    if energy_scan:
        run_energy_scan(
            designer,
            evaluate=args.evaluate,
            pairs_text=args.pairs,
            best_only=args.best,
        )


if __name__ == "__main__":
    main()
'''


def render_offline_script(manifest: dict[str, Any]) -> str:
    """Return a standalone script that reproduces one study outside the web app.

    The result is a single file: the study's parameters as module constants,
    grouped the way :class:`grax.MultilayerDesignConfig` groups them, then both
    stages behind command-line flags. It depends on nothing but ``grax`` -- the
    point is that it can be copied to another machine and run there.
    """

    config = manifest["config"]
    flat = flatten_config_values(config)
    by_section: dict[str, list[str]] = {prefix: [] for prefix, _, _ in _SCRIPT_SECTIONS}
    constructor: list[str] = ["    output_dir=OUTPUT_DIR,"]

    for spec in STUDY_FIELDS:
        prefix = next(
            (key for key, _, _ in _SCRIPT_SECTIONS if spec.section.startswith(key)),
            "Shared",
        )
        block, _, leaf = spec.name.partition(".")
        if leaf:
            continue  # the nested blocks are emitted whole, below
        constant = spec.name.upper()
        by_section[prefix].append(
            f"{constant} = {_script_literal(spec.name, flat.get(spec.name))}"
        )
        constructor.append(f"    {spec.name}={constant},")

    for key, constant in _SCRIPT_SCAN_BLOCKS.items():
        prefix = "Survey" if key.startswith("survey") else "Energy scan"
        settings = dict(config.get(key) or {})
        lines = [f"{constant} = ThetaSearchScanSettings("]
        # Dataclass order (rough, then fine, then final, then peak/roughness),
        # not whatever order the stored JSON happens to be in.
        lines += [
            f"    {name}={settings[name]!r},"
            for name in ThetaSearchScanSettings.__dataclass_fields__
            if name in settings
        ]
        lines.append(")")
        by_section[prefix].append("\n".join(lines))
        constructor.append(f"    {key}={constant},")

    sections: list[str] = []
    for prefix, title, note in _SCRIPT_SECTIONS:
        sections.append(_script_banner(title, note))
        if prefix == "Shared":
            sections.append('OUTPUT_DIR = Path(__file__).resolve().parent / "results"')
        # Scalar assignments run together; a multi-line block gets air around it.
        body: list[str] = []
        for piece in by_section[prefix]:
            if body and ("\n" in piece or "\n" in body[-1]):
                body.append("")
            body.append(piece)
        sections.append("\n".join(body))
    parameters = "\n\n".join(sections)

    return _OFFLINE_SCRIPT_TEMPLATE.format(
        display_name=manifest.get("display_name", "multilayer design"),
        study_id=manifest.get("id", "unknown"),
        script_name=offline_script_filename(manifest),
        generated_at=datetime.now().isoformat(timespec="seconds"),
        parameters=parameters,
        constructor="\n".join(constructor),
    )


def offline_script_filename(manifest: dict[str, Any]) -> str:
    """Return the filename to offer the generated script under."""

    return f"{_slugify(str(manifest.get('id', 'multilayer_design')))}.py".replace("-", "_")


# --------------------------------------------------------------------------- #
# Survey-derived design choices                                               #
# --------------------------------------------------------------------------- #
def survey_design_options(study_dir: Path) -> dict[str, Any]:
    """Return the ``(d, blaze)`` choices offered after a completed survey.

    Reads ``survey/survey.csv`` and mirrors ``1_run_energy_scan.py``'s selection
    logic, so the web offers exactly the designs the CLI would.

    Returns:
        A dict with ``d_values``/``blaze_values`` (sorted grids for the
        dropdowns), ``efficiency`` (``{"d,blaze": value}`` for the JS hint),
        ``best`` (the single global-argmax pair, or ``None``) and ``per_d``
        (one pair per d-spacing). Empty lists when the CSV is missing or holds
        no usable rows.
    """

    import pandas as pd

    empty: dict[str, Any] = {
        "d_values": [],
        "blaze_values": [],
        "efficiency": {},
        "best": None,
        "per_d": [],
    }
    csv_path = Path(study_dir) / "survey" / "survey.csv"
    if not csv_path.is_file():
        return empty
    try:
        table = pd.read_csv(csv_path).dropna(subset=["peak_efficiency"])
    except (OSError, ValueError, KeyError):
        # The survey rewrites this file as it runs; a read that lands badly is
        # answered with "nothing yet" and the next poll picks it up.
        return empty
    if table.empty:
        return empty

    best_row = table.loc[table["peak_efficiency"].idxmax()]
    per_d_rows = table.loc[table.groupby("d_nm")["peak_efficiency"].idxmax()]
    return {
        "d_values": sorted({round(float(v), 3) for v in table["d_nm"]}),
        "blaze_values": sorted({round(float(v), 3) for v in table["blaze_deg"]}),
        "efficiency": {
            f"{float(row.d_nm):.3f},{float(row.blaze_deg):.3f}": float(row.peak_efficiency)
            for row in table.itertuples()
        },
        "best": [float(best_row.d_nm), float(best_row.blaze_deg)],
        "per_d": [
            [float(row.d_nm), float(row.blaze_deg)] for row in per_d_rows.itertuples()
        ],
    }


def resolve_designs(
    mode: str, options: dict[str, Any], raw_designs: list[str]
) -> list[list[float]]:
    """Resolve a selection ``mode`` into the ``(d, blaze)`` pairs to scan.

    Args:
        mode: One of :data:`DESIGN_MODES`.
        options: The dict from :func:`survey_design_options`.
        raw_designs: ``"d,blaze"`` strings, used only when ``mode`` is
            ``"manual"``.

    Returns:
        The pairs to hand to
        :meth:`grax.MultilayerGratingDesigner.run_energy_scan`.

    Raises:
        ValueError: If the mode is unknown, the survey offers nothing, or a
            manual pair is malformed or off the survey grid.
    """

    if mode not in DESIGN_MODES:
        raise ValueError(f"Unknown design mode {mode!r}.")
    if mode == "best":
        if not options.get("best"):
            raise ValueError("The survey has no usable cell to pick a best design from.")
        return [list(options["best"])]
    if mode == "per_d":
        if not options.get("per_d"):
            raise ValueError("The survey has no usable cells to pick designs from.")
        return [list(pair) for pair in options["per_d"]]

    d_values = {round(float(v), 3) for v in options.get("d_values", ())}
    blaze_values = {round(float(v), 3) for v in options.get("blaze_values", ())}
    designs: list[list[float]] = []
    seen: set[tuple[float, float]] = set()
    for raw in raw_designs:
        text = str(raw).strip()
        if not text:
            continue
        try:
            d_text, blaze_text = text.split(",")
            pair = (round(float(d_text), 3), round(float(blaze_text), 3))
        except ValueError as error:
            raise ValueError(f"Could not read the design {text!r}.") from error
        if pair[0] not in d_values or pair[1] not in blaze_values:
            raise ValueError(f"The design {text!r} is not on the survey grid.")
        if pair in seen:
            continue
        seen.add(pair)
        designs.append([pair[0], pair[1]])
    if not designs:
        raise ValueError("Pick at least one (d, blaze) design to scan.")
    return designs


# --------------------------------------------------------------------------- #
# Study store                                                                 #
# --------------------------------------------------------------------------- #
class MultilayerDesignStudyStore:
    """Store multilayer-design study manifests in a filesystem directory."""

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

    def create(
        self,
        *,
        display_name: str,
        config: dict[str, Any],
        auto_energy_scan: str = "none",
    ) -> dict[str, Any]:
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
            "auto_energy_scan": (
                auto_energy_scan if auto_energy_scan in {"best", "none"} else "none"
            ),
            "stages": {stage: _blank_stage() for stage in STAGES},
        }
        return self.save(manifest)

    def delete_many(self, study_ids: list[str]) -> None:
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
        "designs": [],
        "artifacts": {},
    }
