"""Configuration, validation, expansion, and output helpers for custom-grid runs."""

from __future__ import annotations

import csv
import itertools
import math
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from grax.simulation.core import normalize_polarization

from .persistence import build_grating_from_spec

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ParameterDescriptor:
    """One parameter that the custom-grid builder may vary."""

    id: str
    label: str
    category: str
    kind: str
    target: str
    path: tuple[str | int, ...] = ()
    minimum: float | None = None
    choices: tuple[str, ...] = ()
    roughness_path: tuple[str | int, ...] = ()

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-compatible descriptor for the browser builder."""

        payload = asdict(self)
        payload["path"] = list(self.path)
        payload["choices"] = list(self.choices)
        return payload


def parameter_catalog(spec: dict[str, Any]) -> list[ParameterDescriptor]:
    """Return the controlled parameter catalog for one saved grating spec."""

    descriptors = [
        _descriptor("energy_ev", "Energy, eV", "Conditions", "float", "case", minimum=1e-12),
        _descriptor("grazing_angle_deg", "Grazing angle, deg", "Conditions", "float", "case"),
        _descriptor("diffraction_order", "Diffraction order", "Conditions", "int", "case"),
        _descriptor(
            "polarization",
            "Polarization",
            "Conditions",
            "choice",
            "case",
            choices=("s", "p"),
        ),
        _descriptor("fourier_orders", "Fourier orders", "Numerics", "int", "case", minimum=1),
        _descriptor(
            "solver",
            "Solver",
            "Numerics",
            "choice",
            "case",
            choices=("rcwa", "neviere"),
        ),
        _spec_descriptor(
            "period_lpermm", "Line density, lines/mm", "Common geometry", "int", minimum=1
        ),
        _spec_descriptor("x_resolution_nm", "X resolution, nm", "Numerics", "float", minimum=1e-12),
        _spec_descriptor("z_resolution_nm", "Z resolution, nm", "Numerics", "float", minimum=1e-12),
    ]

    grating_type = str(spec.get("grating_type", ""))
    if grating_type == "laminar":
        descriptors.extend(
            [
                _spec_descriptor(
                    "width_to_period_ratio",
                    "Duty cycle",
                    "Laminar geometry",
                    "float",
                    minimum=1e-12,
                ),
                _spec_descriptor(
                    "depth_nm", "Depth, nm", "Laminar geometry", "float", minimum=1e-12
                ),
                _spec_descriptor(
                    "left_wall_angle_deg",
                    "Left wall angle, deg",
                    "Laminar geometry",
                    "float",
                ),
                _spec_descriptor(
                    "right_wall_angle_deg",
                    "Right wall angle, deg",
                    "Laminar geometry",
                    "float",
                ),
            ]
        )
    elif grating_type == "sinusoidal":
        descriptors.append(
            _spec_descriptor("depth_nm", "Depth, nm", "Sinusoidal geometry", "float", minimum=1e-12)
        )
    elif grating_type == "blazed":
        descriptors.extend(
            [
                _spec_descriptor("blaze_angle_deg", "Blaze angle, deg", "Blazed geometry", "float"),
                _spec_descriptor(
                    "anti_blaze_angle_deg",
                    "Anti-blaze angle, deg",
                    "Blazed geometry",
                    "optional_float",
                ),
            ]
        )

    for key, label, kind, minimum in (
        ("kind", "Roughness model", "choice", None),
        ("sigma_nm", "Roughness sigma, nm", "float", 0.0),
        ("correlation_length_nm", "Roughness correlation length, nm", "optional_float", 0.0),
        ("seed", "Roughness seed", "int", 0.0),
        ("num_supercells", "Roughness supercells", "int", 1.0),
        ("num_realizations", "Roughness realizations", "int", 1.0),
    ):
        choices = ("debye-waller", "random-interface") if key == "kind" else ()
        descriptors.append(
            ParameterDescriptor(
                id=f"roughness.{key}",
                label=label,
                category="Roughness",
                kind=kind,
                target="spec",
                path=("roughness", key),
                minimum=minimum,
                choices=choices,
            )
        )

    stack = spec.get("stack")
    if isinstance(stack, dict):
        descriptors.extend(_stack_descriptors(stack))
    return descriptors


def common_parameter_catalog(variants: Iterable[dict[str, Any]]) -> list[ParameterDescriptor]:
    """Return descriptors whose ID and type are valid for every variant."""

    catalogs = [{item.id: item for item in parameter_catalog(spec)} for spec in variants]
    if not catalogs:
        return []
    common_ids = set(catalogs[0])
    for catalog in catalogs[1:]:
        common_ids &= set(catalog)
    first = catalogs[0]
    return [
        item
        for item in first.values()
        if item.id in common_ids
        and all(catalog[item.id].kind == item.kind for catalog in catalogs[1:])
    ]


def default_configuration(base_spec: dict[str, Any]) -> dict[str, Any]:
    """Return the initial one-case custom-grid configuration."""

    return {
        "schema_version": SCHEMA_VERSION,
        "variants": [
            {
                "id": "base",
                "name": str(base_spec.get("name") or "Base grating"),
                "spec": deepcopy(base_spec),
            }
        ],
        "independent": [],
        "coupled_groups": [],
        "display_order": [],
    }


def normalize_configuration(config: dict[str, Any], base_spec: dict[str, Any], *, extra_descriptors: Iterable[ParameterDescriptor] = ()) -> dict[str, Any]:
    """Validate and normalize a submitted custom-grid configuration."""

    if int(config.get("schema_version", SCHEMA_VERSION)) != SCHEMA_VERSION:
        raise ValueError("Unsupported custom-grid schema version.")
    variants_input = config.get("variants") or default_configuration(base_spec)["variants"]
    if not isinstance(variants_input, list) or not variants_input:
        raise ValueError("At least one structural grating variant is required.")

    variants: list[dict[str, Any]] = []
    variant_ids: set[str] = set()
    for index, raw_variant in enumerate(variants_input):
        if not isinstance(raw_variant, dict):
            raise ValueError("Each structural variant must be an object.")
        variant_id = str(raw_variant.get("id") or f"variant-{index + 1}").strip()
        if not variant_id or variant_id in variant_ids:
            raise ValueError("Structural variant IDs must be non-empty and unique.")
        spec = deepcopy(raw_variant.get("spec") or base_spec)
        grating = build_grating_from_spec(spec)
        grating._validate_simulation_materials()
        variants.append(
            {
                "id": variant_id,
                "name": str(raw_variant.get("name") or variant_id).strip(),
                "spec": spec,
            }
        )
        variant_ids.add(variant_id)

    catalog = {item.id: item for item in common_parameter_catalog(v["spec"] for v in variants)}
    catalog.update({item.id: item for item in extra_descriptors})
    variant_descriptor = ParameterDescriptor(
        id="variant",
        label="Structural grating variant",
        category="Structure",
        kind="choice",
        target="variant",
        choices=tuple(v["id"] for v in variants),
    )
    catalog["variant"] = variant_descriptor

    used: set[str] = set()
    independent: list[dict[str, Any]] = []
    for raw_axis in config.get("independent") or []:
        if not isinstance(raw_axis, dict):
            raise ValueError("Independent axes must be objects.")
        parameter = str(raw_axis.get("parameter", ""))
        descriptor = _claim_descriptor(parameter, catalog, used)
        values = _axis_values(raw_axis, descriptor)
        mode = str(raw_axis.get("mode", "list")).strip().lower()
        normalized_axis: dict[str, Any] = {
            "parameter": parameter,
            "mode": mode,
            "values": values,
        }
        if mode == "range":
            normalized_axis.update(
                {
                    "minimum": float(raw_axis["minimum"]),
                    "maximum": float(raw_axis["maximum"]),
                    "count": int(raw_axis["count"]),
                    "step_size": (abs(float(raw_axis["maximum"]) - float(raw_axis["minimum"])) / (int(raw_axis["count"]) - 1)) if int(raw_axis["count"]) > 1 else 0.0,
                }
            )
        independent.append(normalized_axis)

    coupled_groups: list[dict[str, Any]] = []
    for group_index, raw_group in enumerate(config.get("coupled_groups") or []):
        if not isinstance(raw_group, dict):
            raise ValueError("Coupled groups must be objects.")
        parameters = [str(item) for item in raw_group.get("parameters") or []]
        if len(parameters) < 2:
            raise ValueError("Each coupled group needs at least two parameters.")
        descriptors = [_claim_descriptor(parameter, catalog, used) for parameter in parameters]
        raw_rows = raw_group.get("rows") or []
        if not isinstance(raw_rows, list) or not raw_rows:
            raise ValueError("Each coupled group needs at least one row.")
        rows: list[dict[str, Any]] = []
        for raw_row in raw_rows:
            if not isinstance(raw_row, dict) or set(raw_row) != set(parameters):
                raise ValueError("Every coupled row must provide one value for every parameter.")
            rows.append(
                {
                    parameter: _cast_value(raw_row[parameter], descriptor)
                    for parameter, descriptor in zip(parameters, descriptors, strict=True)
                }
            )
        coupled_groups.append(
            {
                "name": str(raw_group.get("name") or f"Coupled group {group_index + 1}"),
                "parameters": parameters,
                "rows": rows,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "variants": variants,
        "independent": independent,
        "coupled_groups": coupled_groups,
        "display_order": [
            *({"kind": "independent", "parameter": axis["parameter"]} for axis in independent),
            *(
                {"kind": "coupled_group", "index": index, "name": group["name"]}
                for index, group in enumerate(coupled_groups)
            ),
        ],
    }


def expand_custom_grid(
    config: dict[str, Any],
    base_spec: dict[str, Any],
    *,
    defaults: dict[str, Any] | None = None,
    extra_descriptors: Iterable[ParameterDescriptor] = (),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Normalize a configuration and expand it into deterministic batch cases."""

    extra_descriptors = tuple(extra_descriptors)
    normalized = normalize_configuration(config, base_spec, extra_descriptors=extra_descriptors)
    variants = {variant["id"]: variant for variant in normalized["variants"]}
    catalog = {
        item.id: item
        for item in common_parameter_catalog(variant["spec"] for variant in normalized["variants"])
    }
    catalog.update({item.id: item for item in extra_descriptors})
    catalog["variant"] = ParameterDescriptor(
        id="variant",
        label="Structural grating variant",
        category="Structure",
        kind="choice",
        target="variant",
        choices=tuple(variants),
    )

    axes: list[list[dict[str, Any]]] = []
    axis_drivers: list[str] = []
    for axis in normalized["independent"]:
        axes.append([{axis["parameter"]: value} for value in axis["values"]])
        axis_drivers.append(axis["parameter"])
    for group in normalized["coupled_groups"]:
        axes.append([dict(row) for row in group["rows"]])
        axis_drivers.append(group["parameters"][0])
    if not axes:
        axes = [[{}]]

    base_defaults = {
        "energy_ev": 100.0,
        "grazing_angle_deg": 1.5,
        "diffraction_order": 1,
        "fourier_orders": 5,
        "polarization": "s",
        "solver": "rcwa",
        **(defaults or {}),
    }
    only_axis_driver = axis_drivers[0] if len(axis_drivers) == 1 else None
    parameter_order = [axis["parameter"] for axis in normalized["independent"]]
    parameter_order.extend(
        parameter for group in normalized["coupled_groups"] for parameter in group["parameters"]
    )

    cases: list[dict[str, Any]] = []
    for index, parts in enumerate(itertools.product(*axes)):
        selected: dict[str, Any] = {}
        for part in parts:
            selected.update(part)
        variant_id = str(selected.get("variant", normalized["variants"][0]["id"]))
        if variant_id not in variants:
            raise ValueError(f"Unknown structural variant: {variant_id}")
        spec = deepcopy(variants[variant_id]["spec"])
        stack = spec.get("stack", {})
        top_side = next((
            key for key in ("material_a", "material_b")
            if _material_name(stack.get(key))
            and _material_name(stack.get(key)) == _material_name(stack.get("top_material"))
        ), None)
        resolved = dict(base_defaults)
        for parameter, value in sorted(
            selected.items(),
            key=lambda item: (catalog[item[0]].kind != "material", len(catalog[item[0]].path)),
        ):
            descriptor = catalog.get(parameter)
            if descriptor is None:
                raise ValueError(f"Parameter is not valid for every selected variant: {parameter}")
            if descriptor.target == "spec":
                if descriptor.kind == "material":
                    material = deepcopy(value)
                    sigma = material.pop("roughness_sigma_nm", None)
                    _set_path(spec, descriptor.path, material)
                    if sigma is not None and descriptor.roughness_path:
                        _set_path(spec, descriptor.roughness_path, sigma)
                else:
                    _set_path(spec, descriptor.path, value)
            elif descriptor.target == "case":
                resolved[parameter] = value

        if top_side and not any(key.startswith("stack.top_material") for key in selected):
            stack["top_material"] = deepcopy(stack[top_side])
        elif top_side:
            matching = next((
                key for key in ("material_a", "material_b")
                if _material_name(stack[key]) == _material_name(stack["top_material"])
            ), None)
            if matching:
                top = stack["top_material"]
                if isinstance(top, dict) and top.get("density_g_cm3") is not None:
                    _set_path(spec, ("stack", matching, "density_g_cm3"), top["density_g_cm3"])
                stack["top_material"] = deepcopy(stack[matching])

        grating = build_grating_from_spec(spec)
        grating._validate_simulation_materials()
        energy = float(resolved["energy_ev"])
        angle = float(resolved["grazing_angle_deg"])
        if not math.isfinite(energy) or energy <= 0:
            raise ValueError("Energy must be finite and greater than 0 eV.")
        if not math.isfinite(angle):
            raise ValueError("Grazing angle must be finite.")
        polarization = normalize_polarization(resolved["polarization"])
        solver = str(resolved["solver"]).strip().lower()
        if solver not in {"rcwa", "neviere"}:
            raise ValueError("Solver must be rcwa or neviere.")

        custom_parameters = {parameter: selected[parameter] for parameter in parameter_order}
        custom_parameters.setdefault("variant", variant_id)
        resolved_parameters: dict[str, Any] = {}
        for parameter, descriptor in catalog.items():
            if descriptor.target == "case":
                resolved_parameters[parameter] = resolved[parameter]
            elif descriptor.target == "spec":
                resolved_parameters[parameter] = deepcopy(_get_path(spec, descriptor.path))
        resolved_parameters["variant"] = variant_id
        resolved_parameter_order = [*catalog]
        plot_value = selected.get(only_axis_driver) if only_axis_driver else index + 1
        plot_numeric = isinstance(plot_value, (int, float)) and not isinstance(plot_value, bool)
        cases.append(
            {
                "case_id": f"custom-grid-{index:08d}",
                "label": f"Custom case {index + 1}",
                "grating": grating,
                "custom_grating_spec": deepcopy(spec),
                "energy_ev": energy,
                "grazing_angle_deg": angle,
                "diffraction_order": int(resolved["diffraction_order"]),
                "fourier_orders": int(resolved["fourier_orders"]),
                "polarization": polarization,
                "solver": solver,
                "x_resolution_nm": float(grating.x_resolution_nm),
                "z_resolution_nm": float(grating.z_resolution_nm),
                "custom_parameters": custom_parameters,
                "custom_parameter_order": parameter_order,
                "custom_resolved_parameters": resolved_parameters,
                "custom_resolved_parameter_order": resolved_parameter_order,
                "custom_variant_id": variant_id,
                "custom_variant_name": variants[variant_id]["name"],
                "custom_plot_x_key": only_axis_driver if plot_numeric else "index",
                "custom_plot_x_value": float(plot_value) if plot_numeric else float(index + 1),
            }
        )
    return normalized, cases


def custom_grid_case_count(config: dict[str, Any], base_spec: dict[str, Any]) -> int:
    """Return the validated number of generated cases."""

    _, cases = expand_custom_grid(config, base_spec)
    return len(cases)


def write_custom_grid_summary(results: Iterable[Any], output_path: str | Path) -> None:
    """Write one custom-grid row per case with all selected parameters."""

    results = list(results)
    parameter_order = _result_parameter_order(results)
    with Path(output_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case_id",
                *parameter_order,
                "selected_efficiency",
                "selected_diffraction_angle_deg",
                "status",
                "error_message",
            ]
        )
        for result in results:
            parameters = result.case_data.get("custom_resolved_parameters") or result.case_data.get(
                "custom_parameters", {}
            )
            writer.writerow(
                [
                    result.case_id,
                    *[_cell(parameters.get(parameter)) for parameter in parameter_order],
                    result.selected_efficiency,
                    result.selected_diffraction_angle_deg,
                    result.status,
                    result.error_message or "",
                ]
            )


def write_custom_grid_all_orders(results: Iterable[Any], output_path: str | Path) -> None:
    """Write all diffraction orders with custom-grid parameter columns."""

    results = list(results)
    parameter_order = _result_parameter_order(results)
    with Path(output_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case_id",
                *parameter_order,
                "energy_ev",
                "grazing_angle_deg",
                "order",
                "efficiency",
                "diffraction_angle_deg",
            ]
        )
        for result in results:
            if result.status != "ok":
                continue
            parameters = result.case_data.get("custom_resolved_parameters") or result.case_data.get(
                "custom_parameters", {}
            )
            for order, efficiency, angle in zip(
                np.asarray(result.orders),
                np.asarray(result.efficiency_all),
                np.asarray(result.diffraction_angle_all),
                strict=True,
            ):
                order_value = float(order)
                writer.writerow(
                    [
                        result.case_id,
                        *[_cell(parameters.get(parameter)) for parameter in parameter_order],
                        result.energy_ev,
                        result.grazing_angle_deg,
                        int(order_value) if order_value.is_integer() else order_value,
                        float(efficiency),
                        float(angle),
                    ]
                )


def plot_custom_grid(results: Iterable[Any], output_path: str | Path, *, title: str) -> None:
    """Plot selected efficiency against the smart custom-grid X coordinate."""

    import matplotlib.pyplot as plt

    successful = [result for result in results if result.status == "ok"]
    if not successful:
        return
    key = str(successful[0].case_data.get("custom_plot_x_key", "index"))
    ordered = sorted(
        successful,
        key=lambda result: float(result.case_data.get("custom_plot_x_value", result.index + 1)),
    )
    x_values = [
        float(result.case_data.get("custom_plot_x_value", result.index + 1)) for result in ordered
    ]
    y_values = [float(result.selected_efficiency) for result in ordered]
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(x_values, y_values, "o-", linewidth=1.2, markersize=4)
    axis.set_xlabel("Case index" if key == "index" else key.replace("_", " "))
    axis.set_ylabel("Selected-order efficiency")
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.relim()
    axis.autoscale_view()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _descriptor(
    identifier: str,
    label: str,
    category: str,
    kind: str,
    target: str,
    *,
    minimum: float | None = None,
    choices: tuple[str, ...] = (),
) -> ParameterDescriptor:
    return ParameterDescriptor(
        id=identifier,
        label=label,
        category=category,
        kind=kind,
        target=target,
        minimum=minimum,
        choices=choices,
    )


def _spec_descriptor(
    key: str,
    label: str,
    category: str,
    kind: str,
    *,
    minimum: float | None = None,
) -> ParameterDescriptor:
    return ParameterDescriptor(
        id=key,
        label=label,
        category=category,
        kind=kind,
        target="spec",
        path=(key,),
        minimum=minimum,
    )


def _stack_descriptors(stack: dict[str, Any]) -> list[ParameterDescriptor]:
    descriptors: list[ParameterDescriptor] = []
    material_keys = {
        "substrate_material",
        "layer_material",
        "material_a",
        "material_b",
        "top_material",
        "top_cap_material",
    }

    def add_value(path: tuple[str | int, ...], value: Any, label: str) -> None:
        identifier = ".".join(str(part) for part in path)
        leaf = str(path[-1])
        if leaf in material_keys or leaf == "material":
            kind = "material"
        elif isinstance(value, bool):
            kind = "choice"
        elif isinstance(value, int) and not isinstance(value, bool):
            kind = "int"
        elif isinstance(value, (float, int)):
            kind = "float"
        elif value is None and (leaf.endswith("material") or leaf == "material"):
            kind = "material"
        elif value is None:
            kind = "optional_float"
        else:
            return
        minimum = (
            0.0
            if any(token in leaf for token in ("thickness", "roughness", "correlation"))
            else None
        )
        if leaf == "n_bilayers":
            minimum = 1
        roughness_key = {
            "substrate_material": "substrate_roughness_sigma_nm",
            "layer_material": "layer_roughness_sigma_nm",
            "material_a": "material_a_roughness_sigma_nm",
            "material_b": "material_b_roughness_sigma_nm",
            "top_cap_material": "top_cap_roughness_sigma_nm",
            "material": "roughness_sigma_nm",
        }.get(leaf)
        descriptors.append(
            ParameterDescriptor(
                id=identifier,
                label=label,
                category="Coating",
                kind=kind,
                target="spec",
                path=path,
                minimum=minimum,
                roughness_path=(
                    (*path[:-1], roughness_key) if kind == "material" and roughness_key else ()
                ),
            )
        )
        if kind == "material":
            descriptors.append(
                ParameterDescriptor(
                    id=f"{identifier}.density_g_cm3",
                    label=f"{label} density, g/cm³",
                    category="Coating",
                    kind="optional_float",
                    target="spec",
                    path=(*path, "density_g_cm3"),
                    minimum=1e-12,
                )
            )

    for key, value in stack.items():
        if key == "type":
            continue
        if key == "layers_bottom_up" and isinstance(value, list):
            for index, layer in enumerate(value):
                if not isinstance(layer, dict):
                    continue
                for layer_key, layer_value in layer.items():
                    add_value(
                        ("stack", key, index, layer_key),
                        layer_value,
                        f"Custom layer {index + 1} {layer_key.replace('_', ' ')}",
                    )
            continue
        add_value(("stack", key), value, key.replace("_", " ").capitalize())
    # Older saved gratings may omit optional interface roughness fields entirely.
    known = {item.id for item in descriptors}
    for material in list(descriptors):
        if material.roughness_path:
            identifier = ".".join(str(part) for part in material.roughness_path)
            if identifier not in known:
                descriptors.append(ParameterDescriptor(
                    id=identifier, label=f"{material.label} roughness σ, nm",
                    category="Coating", kind="optional_float", target="spec",
                    path=material.roughness_path, minimum=0.0,
                ))
                known.add(identifier)
    return descriptors


def _claim_descriptor(
    parameter: str,
    catalog: dict[str, ParameterDescriptor],
    used: set[str],
) -> ParameterDescriptor:
    if parameter not in catalog:
        raise ValueError(f"Unknown or incompatible custom-grid parameter: {parameter}")
    if parameter in used:
        raise ValueError(f"Parameter may only appear once in a custom grid: {parameter}")
    used.add(parameter)
    return catalog[parameter]


def _axis_values(raw_axis: dict[str, Any], descriptor: ParameterDescriptor) -> list[Any]:
    mode = str(raw_axis.get("mode", "list")).strip().lower()
    if mode == "range":
        if descriptor.kind not in {"float", "int", "optional_float"}:
            raise ValueError(f"{descriptor.label} only supports explicit values.")
        lower = float(raw_axis["minimum"])
        upper = float(raw_axis["maximum"])
        count = int(raw_axis["count"])
        if count < 1 or not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError("Range bounds must be finite and the count must be at least 1.")
        values: list[Any] = np.linspace(lower, upper, count).tolist()
        if descriptor.kind == "int":
            values = list(dict.fromkeys(int(np.rint(value)) for value in values))
    elif mode == "list":
        raw_values = raw_axis.get("values")
        if not isinstance(raw_values, list) or not raw_values:
            raise ValueError(f"{descriptor.label} needs at least one explicit value.")
        values = raw_values
    else:
        raise ValueError("Axis mode must be list or range.")
    return [_cast_value(value, descriptor) for value in values]


def _cast_value(value: Any, descriptor: ParameterDescriptor) -> Any:
    if descriptor.kind == "float":
        result: Any = float(value)
    elif descriptor.kind == "optional_float":
        result = None if value in (None, "") else float(value)
    elif descriptor.kind == "int":
        number = float(value)
        if not number.is_integer():
            raise ValueError(f"{descriptor.label} only accepts integers.")
        result = int(number)
    elif descriptor.kind == "choice":
        result = str(value)
        if descriptor.choices and result not in descriptor.choices:
            raise ValueError(f"Invalid value for {descriptor.label}: {result}")
    elif descriptor.kind == "material":
        if isinstance(value, dict):
            result = deepcopy(value)
        else:
            result = {"name": str(value).strip(), "density_g_cm3": None}
        if not result.get("name"):
            raise ValueError(f"{descriptor.label} needs a material name.")
        for key, positive in (("density_g_cm3", True), ("roughness_sigma_nm", False)):
            raw = result.get(key)
            if raw in (None, ""):
                result[key] = None
                continue
            number = float(raw)
            if not math.isfinite(number) or (number <= 0 if positive else number < 0):
                constraint = "positive" if positive else "non-negative"
                raise ValueError(f"{descriptor.label} {key} must be finite and {constraint}.")
            if key == "roughness_sigma_nm" and not descriptor.roughness_path:
                raise ValueError(f"{descriptor.label} has no separate interface roughness setting.")
            result[key] = number
    else:
        raise ValueError(f"Unsupported parameter type: {descriptor.kind}")
    if isinstance(result, (float, int)) and descriptor.minimum is not None:
        if result < descriptor.minimum:
            raise ValueError(f"{descriptor.label} must be at least {descriptor.minimum}.")
    if isinstance(result, float) and not math.isfinite(result):
        raise ValueError(f"{descriptor.label} must be finite.")
    return result


def _material_name(value: Any) -> str | None:
    if isinstance(value, dict):
        return value.get("name")
    return value if isinstance(value, str) else None


def _set_path(root: dict[str, Any], path: tuple[str | int, ...], value: Any) -> None:
    target: Any = root
    for index, part in enumerate(path[:-1]):
        if path[index + 1] == "density_g_cm3":
            if isinstance(target[part], str):
                target[part] = {"name": target[part], "density_g_cm3": None}
            elif target[part] is None:
                raise ValueError("Choose a material before scanning its density.")
        if isinstance(target, dict) and target.get(part) is None and part == "roughness":
            target[part] = {
                "kind": "debye-waller",
                "sigma_nm": 0.0,
                "seed": 0,
                "resolution_factor": 4.0,
                "correlation_length_nm": None,
                "num_supercells": 1,
                "num_realizations": 1,
            }
        target = target[part]
    target[path[-1]] = deepcopy(value)


def _get_path(root: dict[str, Any], path: tuple[str | int, ...]) -> Any:
    """Read a registered nested path from a grating specification."""
    target: Any = root
    for part in path:
        if target is None:
            return None
        if isinstance(target, dict) and part not in target:
            return None
        if isinstance(target, str) and part == "density_g_cm3":
            return None
        target = target[part]
    return target


def _result_parameter_order(results: list[Any]) -> list[str]:
    order: list[str] = []
    for result in results:
        requested_order = result.case_data.get(
            "custom_resolved_parameter_order"
        ) or result.case_data.get("custom_parameter_order", [])
        parameters = result.case_data.get("custom_resolved_parameters") or result.case_data.get(
            "custom_parameters", {}
        )
        for parameter in requested_order:
            if parameter not in order:
                order.append(parameter)
        for parameter in parameters:
            if parameter not in order:
                order.append(parameter)
    return order


def _cell(value: Any) -> Any:
    if isinstance(value, dict) and "name" in value:
        density = value.get("density_g_cm3")
        return value["name"] if density is None else f"{value['name']} ({density} g/cm3)"
    return value


__all__ = [
    "SCHEMA_VERSION",
    "ParameterDescriptor",
    "common_parameter_catalog",
    "custom_grid_case_count",
    "default_configuration",
    "expand_custom_grid",
    "normalize_configuration",
    "parameter_catalog",
    "plot_custom_grid",
    "write_custom_grid_all_orders",
    "write_custom_grid_summary",
]
