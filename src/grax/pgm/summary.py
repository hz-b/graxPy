"""Title and parameter text of the SX700 dashboard."""

from __future__ import annotations

from .geometry import calculate_touch_diff_deg
from .grating import (
    _grating_short_description,
    _grating_text_lines,
    _material_label_text,
)
from .parameters import SX700Parameters


def _is_multilayer(stack) -> bool:
    return hasattr(stack, "material_a") and hasattr(stack, "n_bilayers")


def _mirror_short_description(stack) -> str:
    if stack is None:
        return "Mirror"
    if _is_multilayer(stack):
        return (
            f"Mirror {_material_label_text(stack.material_a)}/"
            f"{_material_label_text(stack.material_b)} ML"
        )
    layers = [_material_label_text(material) for material, _ in stack.layer_sequence_bottom_up()]
    names = "/".join(reversed(layers)) or _material_label_text(stack.substrate_material)
    return f"Mirror {names}"


def _mirror_text_lines(stack) -> list[str]:
    if stack is None:
        return ["Mirror             (none)"]
    if _is_multilayer(stack):
        lines = [
            "Mirror model       multilayer",
            f"Mirror ML          {_material_label_text(stack.material_a)}/"
            f"{_material_label_text(stack.material_b)}",
            f"Mirror ML d        {float(stack.material_a_thickness_nm):g}/"
            f"{float(stack.material_b_thickness_nm):g} nm",
            f"Mirror ML periods  {int(stack.n_bilayers):d}",
        ]
    else:
        lines = ["Mirror model       layers"]
        for material, thickness in reversed(list(stack.layer_sequence_bottom_up())):
            lines.append(f"Layer              {_material_label_text(material)} / {thickness:g} nm")
    lines.append(f"Mirror substrate   {_material_label_text(stack.substrate_material)}")
    return lines


def parameter_blocks(p: SX700Parameters) -> tuple[str, str]:
    """Return the SX700 and optics text blocks of the dashboard."""
    sx700 = [
        "SX700 PARAMETERS",
        "────────────────────",
        f"Premirror L        {p.mirror_length_mm:g} mm",
        f"Grating L          {p.grating_length_mm:g} mm",
        f"Optics distance x  {p.optic_dx_mm:g} mm",
        f"Beam offset b      {p.beam_offset_mm:g} mm",
        f"Axis offset v      {p.mirror_grating_axis_dy_mm:g} mm",
        f"Offset c           {p.c_offset_mm:g} mm",
        f"Offset a           {p.a_offset_mm:g} mm",
        "",
        f"Theta min/max      {p.theta_min_deg:g}° / {p.theta_max_deg:g}°",
        f"|Beta| min/max     {p.beta_min_deg:g}° / {p.beta_max_deg:g}°",
        "",
        f"Beam height        {p.beam_height_mm:g} mm",
        f"Groove density     {p.groove_density_l_mm:g} l/mm",
        f"Diffraction order  {int(p.diffraction_order):d}",
        f"Edge margin        {p.edge_margin_mm:g} mm",
        f"Critical distance  {p.crit_distance:g} mm",
        f"touch_diff         {calculate_touch_diff_deg(p):.2f}°",
        "",
        f"Energy range       {p.energy_min_eV:g}–{p.energy_max_eV:g} eV",
        f"c_ff range         {p.c_min:g}–{p.c_max:g}",
    ]
    optics = [
        "OPTICS",
        "────────────────────",
        *_mirror_text_lines(p.mirror_stack),
        "",
        *_grating_text_lines(p.grating),
        f"Polarization       {p.polarization}",
        f"RCWA solver        {p.rcwa_solver}",
        f"RCWA orders        {2 * p.rcwa_harmonics + 1}",
        f"RCWA adaptive      {p.rcwa_adaptive_sampling}",
    ]
    return "\n".join(sx700), "\n".join(optics)


def dashboard_title(p: SX700Parameters) -> str:
    """Return the dashboard's title line with the active optics."""
    return (
        "SX700 / PGM – Geometry and efficiency"
        f"   |   {p.groove_density_l_mm:g} l/mm"
        f"   |   {_mirror_short_description(p.mirror_stack)}"
        f"   |   {_grating_short_description(p.grating)}"
    )
