"""File-based grating persistence for the local web app."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import asdict
import math
from grax.roughness import RoughnessSpec
from datetime import datetime
from pathlib import Path
from typing import Any

from grax.gratings import (
    BaseGrating,
    BlazedGrating,
    LaminarGrating,
    SinusoidalGrating,
)
from grax.afm_grating import AFMGrating
from grax.materials import MaterialSpec, material_density_g_cm3, material_label, validate_material_input
from grax.stacks import (
    BareStack, BaseStack, CustomStack, LayerSpec, MultilayerStack, SingleLayerStack,
)

from .materials import OpticalConstantsTable, load_material_catalog

SCHEMA_VERSION = 4


class GratingStore:
    """Store saved grating specs as individual JSON files."""

    def __init__(self, directory: str | Path) -> None:
        """Initialize the store.

        Args:
            directory: Directory containing saved grating JSON files.
        """
        self.directory = Path(directory)

    def list(self) -> list[dict[str, Any]]:
        """Return saved grating specs ordered by name."""
        if not self.directory.exists():
            return []
        specs = [self.load(path.stem) for path in self.directory.glob("*.json")]
        return sorted(specs, key=lambda spec: str(spec.get("name", "")).lower())

    def load(self, grating_id: str) -> dict[str, Any]:
        """Load one grating spec by ID."""
        path = self._path_for_id(grating_id)
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def save(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Save a grating spec and return the persisted payload."""
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = deepcopy(spec)
        now = datetime.now().isoformat(timespec="seconds")
        payload["schema_version"] = SCHEMA_VERSION
        payload.setdefault("id", self._unique_id(str(payload.get("name", "grating"))))
        payload.setdefault("created_at", now)
        payload["updated_at"] = now
        path = self._path_for_id(str(payload["id"]))
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return payload

    def delete(self, grating_id: str) -> None:
        """Delete one saved grating by ID if it exists."""

        path = self._path_for_id(grating_id)
        if path.exists():
            path.unlink()

    def _path_for_id(self, grating_id: str) -> Path:
        """Return the JSON path for one grating ID."""
        safe_id = _slugify(grating_id)
        if safe_id != grating_id:
            raise ValueError("Invalid grating id.")
        return self.directory / f"{safe_id}.json"

    def _unique_id(self, name: str) -> str:
        """Return a unique store ID derived from a display name."""
        base = _slugify(name) or "grating"
        candidate = base
        index = 2
        while (self.directory / f"{candidate}.json").exists():
            candidate = f"{base}-{index}"
            index += 1
        return candidate


def grating_to_spec(grating: BaseGrating, *, name: str) -> dict[str, Any]:
    """Convert a supported grating object into a JSON-compatible spec."""
    common = {
        "name": name,
        "period_lpermm": grating.period_lpermm,
        "x_resolution_nm": grating.x_resolution_nm,
        "z_resolution_nm": grating.z_resolution_nm,
        "stack": stack_to_spec(grating.resolved_stack()),
        "roughness": None if grating.roughness is None else asdict(grating.roughness),
    }
    if isinstance(grating, LaminarGrating):
        return {
            **common,
            "grating_type": "laminar",
            "width_to_period_ratio": grating.width_to_period_ratio,
            "depth_nm": grating.depth_nm,
            "left_wall_angle_deg": grating.left_wall_angle_deg,
            "right_wall_angle_deg": grating.right_wall_angle_deg,
        }
    if isinstance(grating, SinusoidalGrating):
        return {
            **common,
            "grating_type": "sinusoidal",
            "depth_nm": grating.depth_nm,
        }
    if isinstance(grating, BlazedGrating):
        return {
            **common,
            "grating_type": "blazed",
            "blaze_angle_deg": grating.blaze_angle_deg,
            "anti_blaze_angle_deg": grating.anti_blaze_angle_deg,
        }
    raise TypeError("Unsupported grating class.")


class PlaneMirrorStore(GratingStore):
    """Store saved plane mirrors, each with its attached scans, as JSON files.

    A plane-mirror spec holds a ``name``, the submitted stack form fields
    (``stack_form``) and a list of ``scans``; every scan has an ``id``, a
    ``name`` and its scan form fields (``form``).
    """


def build_stack_from_spec(stack_spec: dict[str, Any]) -> BaseStack:
    """Restore a stack including optional interface correlation lengths."""
    stack = _build_stack_from_spec(stack_spec)
    for prefix in ("substrate", "top_cap", "layer", "material_a", "material_b"):
        key = prefix + "_correlation_length_nm"
        if hasattr(stack, key):
            setattr(stack, key, _optional_sigma(stack_spec.get(key)))
    return stack


def _build_stack_from_spec(stack_spec: dict[str, Any]) -> BaseStack:
    """Build a coating stack from a JSON-compatible stack spec."""
    stack_spec = dict(stack_spec)
    stack_type = str(stack_spec.get("type", "single_layer"))
    if stack_type == "bare":
        return BareStack(
            substrate_material=_material(
                stack_spec["substrate_material"], field_name="substrate_material"
            ),
            substrate_roughness_sigma_nm=_optional_sigma(
                stack_spec.get("substrate_roughness_sigma_nm")
            ),
        )
    if stack_type == "custom":
        layers = [
            LayerSpec(
                material=_material(
                    layer["material"], field_name=f"layers_bottom_up[{index}].material"
                ),
                thickness_nm=float(layer["thickness_nm"]),
                roughness_sigma_nm=_optional_sigma(layer.get("roughness_sigma_nm")),
                correlation_length_nm=_optional_sigma(layer.get("correlation_length_nm")),
            )
            for index, layer in enumerate(stack_spec["layers_bottom_up"])
        ]
        return CustomStack(
            substrate_material=_material(
                stack_spec["substrate_material"], field_name="substrate_material"
            ),
            layers_bottom_up=layers,
            top_cap_material=_optional_material(
                stack_spec.get("top_cap_material"), field_name="top_cap_material"
            ),
            top_cap_thickness_nm=float(stack_spec.get("top_cap_thickness_nm", 0.0)),
            substrate_roughness_sigma_nm=_optional_sigma(
                stack_spec.get("substrate_roughness_sigma_nm")
            ),
            top_cap_roughness_sigma_nm=_optional_sigma(
                stack_spec.get("top_cap_roughness_sigma_nm")
            ),
        )
    if stack_type == "multilayer":
        return MultilayerStack(
            substrate_material=_material(stack_spec["substrate_material"], field_name="substrate_material"),
            material_a=_material(stack_spec["material_a"], field_name="material_a"),
            material_b=_material(stack_spec["material_b"], field_name="material_b"),
            d_period_nm=float(stack_spec["d_period_nm"]),
            gamma=float(stack_spec["gamma"]),
            n_bilayers=int(stack_spec["n_bilayers"]),
            top_material=_material(stack_spec["top_material"], field_name="top_material"),
            top_cap_material=_optional_material(stack_spec.get("top_cap_material"), field_name="top_cap_material"),
            top_cap_thickness_nm=float(stack_spec.get("top_cap_thickness_nm", 0.0)),
            substrate_roughness_sigma_nm=_optional_sigma(stack_spec.get("substrate_roughness_sigma_nm")),
            material_a_roughness_sigma_nm=_optional_sigma(stack_spec.get("material_a_roughness_sigma_nm")),
            material_b_roughness_sigma_nm=_optional_sigma(stack_spec.get("material_b_roughness_sigma_nm")),
            top_cap_roughness_sigma_nm=_optional_sigma(stack_spec.get("top_cap_roughness_sigma_nm")),
        )
    if stack_type != "single_layer":
        raise ValueError("Unsupported stack type.")
    return SingleLayerStack(
        substrate_material=_material(stack_spec["substrate_material"], field_name="substrate_material"),
        layer_material=_material(stack_spec["layer_material"], field_name="layer_material"),
        layer_thickness_nm=float(stack_spec["layer_thickness_nm"]),
        top_cap_material=_optional_material(stack_spec.get("top_cap_material"), field_name="top_cap_material"),
        top_cap_thickness_nm=float(stack_spec.get("top_cap_thickness_nm", 0.0)),
        substrate_roughness_sigma_nm=_optional_sigma(stack_spec.get("substrate_roughness_sigma_nm")),
        layer_roughness_sigma_nm=_optional_sigma(stack_spec.get("layer_roughness_sigma_nm")),
        top_cap_roughness_sigma_nm=_optional_sigma(stack_spec.get("top_cap_roughness_sigma_nm")),
    )


def build_grating_from_spec(spec: dict[str, Any], catalog=None) -> BaseGrating:
    """Restore a grating and record whether it owns its roughness configuration."""
    grating = _build_grating_from_spec(spec, catalog)
    grating._saved_roughness_configuration = "roughness" in spec
    return grating


def _build_grating_from_spec(
    spec: dict[str, Any],
    catalog: dict[str, OpticalConstantsTable] | None = None,
) -> BaseGrating:
    """Build a supported grating from a saved JSON-compatible spec."""
    stack_spec = dict(spec["stack"])
    common = {
        "period_lpermm": int(spec["period_lpermm"]),
        "x_resolution_nm": float(spec["x_resolution_nm"]),
        "z_resolution_nm": float(spec["z_resolution_nm"]),
        "roughness": RoughnessSpec(**spec["roughness"]) if spec.get("roughness") else None,
    }
    stack = build_stack_from_spec(stack_spec)
    common["substrate_material"] = stack.substrate_material
    if not isinstance(stack, SingleLayerStack):
        common["coating_stack"] = stack
    else:
        # Keep the individual grating fields (for direct attribute reads) and
        # also attach a coating stack so per-interface roughness is carried.
        common.update(
            {
                "substrate_material": stack.substrate_material,
                "layer_material": stack.layer_material,
                "layer_thickness_nm": stack.layer_thickness_nm,
                "top_cap_material": stack.top_cap_material,
                "top_cap_thickness_nm": stack.top_cap_thickness_nm,
                "coating_stack": stack,
            }
        )

    if spec["grating_type"] == "laminar":
        return LaminarGrating(
            **common,
            width_to_period_ratio=float(spec["width_to_period_ratio"]),
            depth_nm=float(spec["depth_nm"]),
            left_wall_angle_deg=float(spec["left_wall_angle_deg"]),
            right_wall_angle_deg=float(spec["right_wall_angle_deg"]),
        )
    if spec["grating_type"] == "afm":
        profile = spec.get("profile") or {}
        profile_path = Path(str(profile.get("profile_path", "")))
        if not profile_path.is_file():
            raise ValueError("AFM profile artifact is missing.")
        values = __import__("numpy").loadtxt(profile_path, delimiter=",", skiprows=1)
        return AFMGrating(
            **common,
            x_points_nm=values[:, 0],
            z_points_nm=values[:, 1],
        )
    if spec["grating_type"] == "sinusoidal":
        return SinusoidalGrating(
            **common,
            depth_nm=float(spec["depth_nm"]),
        )
    if spec["grating_type"] == "blazed":
        anti_blaze = spec.get("anti_blaze_angle_deg")
        return BlazedGrating(
            **common,
            blaze_angle_deg=float(spec["blaze_angle_deg"]),
            anti_blaze_angle_deg=None if anti_blaze in (None, "") else float(anti_blaze),
        )
    raise ValueError("Unsupported grating_type.")


def stack_to_spec(stack: BaseStack) -> dict[str, Any]:
    """Serialize interface parameters while retaining the existing stack format."""
    spec = _stack_to_spec(stack)
    for prefix in ("substrate", "top_cap", "layer", "material_a", "material_b"):
        key = prefix + "_correlation_length_nm"
        if hasattr(stack, key):
            spec[key] = getattr(stack, key)
    if isinstance(stack, CustomStack):
        for layer, saved in zip(stack.layers_bottom_up, spec["layers_bottom_up"], strict=True):
            saved["correlation_length_nm"] = layer.correlation_length_nm
    return spec


def _stack_to_spec(stack: BaseStack) -> dict[str, Any]:
    """Convert a coating stack to a JSON-compatible spec."""
    if isinstance(stack, BareStack):
        return {
            "type": "bare",
            "substrate_material": _material_to_spec(stack.substrate_material),
            "substrate_roughness_sigma_nm": stack.substrate_roughness_sigma_nm,
        }
    if isinstance(stack, CustomStack):
        return {
            "type": "custom",
            "substrate_material": _material_to_spec(stack.substrate_material),
            "layers_bottom_up": [
                {
                    "material": _material_to_spec(layer.material),
                    "thickness_nm": layer.thickness_nm,
                    "roughness_sigma_nm": layer.roughness_sigma_nm,
                }
                for layer in stack.layers_bottom_up
            ],
            "top_cap_material": _optional_material_to_spec(stack.top_cap_material),
            "top_cap_thickness_nm": stack.top_cap_thickness_nm,
            "substrate_roughness_sigma_nm": stack.substrate_roughness_sigma_nm,
            "top_cap_roughness_sigma_nm": stack.top_cap_roughness_sigma_nm,
        }
    if isinstance(stack, MultilayerStack):
        return {
            "type": "multilayer",
            "substrate_material": _material_to_spec(stack.substrate_material),
            "material_a": _material_to_spec(stack.material_a),
            "material_b": _material_to_spec(stack.material_b),
            "d_period_nm": stack.d_period_nm,
            "gamma": stack.gamma,
            "n_bilayers": stack.n_bilayers,
            "top_material": _material_to_spec(stack.top_material),
            "top_cap_material": _optional_material_to_spec(stack.top_cap_material),
            "top_cap_thickness_nm": stack.top_cap_thickness_nm,
            "substrate_roughness_sigma_nm": stack.substrate_roughness_sigma_nm,
            "material_a_roughness_sigma_nm": stack.material_a_roughness_sigma_nm,
            "material_b_roughness_sigma_nm": stack.material_b_roughness_sigma_nm,
            "top_cap_roughness_sigma_nm": stack.top_cap_roughness_sigma_nm,
        }
    if not isinstance(stack, SingleLayerStack):
        raise TypeError("Unsupported stack class.")
    return {
        "type": "single_layer",
        "substrate_material": _material_to_spec(stack.substrate_material),
        "layer_material": _material_to_spec(stack.layer_material),
        "layer_thickness_nm": stack.layer_thickness_nm,
        "top_cap_material": _optional_material_to_spec(stack.top_cap_material),
        "top_cap_thickness_nm": stack.top_cap_thickness_nm,
        "substrate_roughness_sigma_nm": stack.substrate_roughness_sigma_nm,
        "layer_roughness_sigma_nm": stack.layer_roughness_sigma_nm,
        "top_cap_roughness_sigma_nm": stack.top_cap_roughness_sigma_nm,
    }


def _material(key: Any, *, field_name: str) -> Any:
    """Return one validated material from a serialized spec value."""
    material = _material_from_spec_value(key)
    validate_material_input(material, field_name=field_name)
    return material


def _optional_material(
    key: Any,
    *,
    field_name: str,
) -> Any | None:
    """Return an optional validated material from a serialized spec value."""
    if key in (None, ""):
        return None
    return _material(key, field_name=field_name)


def _optional_sigma(value: Any) -> float | None:
    """Return an optional roughness sigma from a serialized spec value."""
    if value in (None, ""):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError("Interface roughness and correlation lengths must be finite and at least 0 nm.")
    return number


def _material_from_spec_value(value: Any) -> Any:
    """Convert one serialized material spec to a runtime material object."""
    if isinstance(value, MaterialSpec):
        return value
    if isinstance(value, dict):
        name = value.get("name")
        if name in (None, ""):
            raise ValueError("Material specs must include a name.")
        density_value = value.get("density_g_cm3")
        density = None if density_value in (None, "") else float(density_value)
        return MaterialSpec(str(name), density)
    if isinstance(value, str):
        return MaterialSpec(value, None)
    return value


def _material_to_spec(material: Any) -> dict[str, Any]:
    """Return a JSON-compatible material spec."""
    return {
        "name": material_label(material),
        "density_g_cm3": material_density_g_cm3(material),
    }


def _optional_material_to_spec(material: Any) -> dict[str, Any] | None:
    """Return an optional JSON-compatible material spec."""
    if material is None:
        return None
    return _material_to_spec(material)


def _slugify(value: str) -> str:
    """Return a filesystem-safe identifier."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return re.sub(r"-{2,}", "-", slug)


__all__ = [
    "GratingStore",
    "PlaneMirrorStore",
    "build_grating_from_spec",
    "build_stack_from_spec",
    "grating_to_spec",
    "stack_to_spec",
    "load_material_catalog",
]
