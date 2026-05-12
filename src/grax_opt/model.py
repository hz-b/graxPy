"""Parameter mapping between Ax and grax gratings."""

from __future__ import annotations

from collections.abc import Mapping

from grax import BlazedGrating, LaminarGrating

from .config import (
    BlazedAxConfig,
    InitialBlazedGrating,
    InitialLaminarGrating,
    LaminarAxConfig,
    ParameterBounds,
)

TOP_CAP_DEFAULT_UPPER_NM = 5.0


def _default_relative_bounds(initial_value: float, relative_fraction: float) -> ParameterBounds:
    """Return symmetric relative bounds around an initial value."""

    lower = initial_value * (1.0 - relative_fraction)
    upper = initial_value * (1.0 + relative_fraction)
    return ParameterBounds(lower=lower, upper=upper)


def _top_cap_default_bounds(initial_value: float) -> ParameterBounds:
    """Return default top-cap bounds when optimization is enabled."""

    if initial_value > 0.0:
        return _default_relative_bounds(initial_value, 0.2)
    return ParameterBounds(lower=0.0, upper=TOP_CAP_DEFAULT_UPPER_NM)


def _resolve_bounds(
    *,
    explicit_bounds: ParameterBounds | None,
    initial_value: float,
    default_relative_fraction: float | None,
    zero_default_factory: callable | None = None,
) -> ParameterBounds:
    """Resolve explicit or default bounds for one parameter."""

    if explicit_bounds is not None:
        return explicit_bounds
    if default_relative_fraction is not None:
        if initial_value == 0.0 and zero_default_factory is not None:
            return zero_default_factory(initial_value)
        return _default_relative_bounds(initial_value, default_relative_fraction)
    raise ValueError("No bounds available for the requested parameter.")


def _range_parameter(name: str, bounds: ParameterBounds) -> dict[str, object]:
    """Return an Ax range parameter definition."""

    return {
        "name": name,
        "type": "range",
        "bounds": bounds.as_list(),
        "value_type": "float",
    }


def build_blazed_ax_parameters(config: BlazedAxConfig) -> list[dict[str, object]]:
    """Build Ax parameter definitions from optimizer config."""

    initial = config.initial_grating
    parameters: list[dict[str, object]] = []

    period_bounds = _resolve_bounds(
        explicit_bounds=config.period_lpermm_bounds,
        initial_value=initial.period_lpermm,
        default_relative_fraction=0.01,
    )
    parameters.append(_range_parameter("period_lpermm", period_bounds))

    if config.optimize_blaze_angle_deg:
        blaze_bounds = _resolve_bounds(
            explicit_bounds=config.blaze_angle_deg_bounds,
            initial_value=initial.blaze_angle_deg,
            default_relative_fraction=0.2,
        )
        parameters.append(_range_parameter("blaze_angle_deg", blaze_bounds))

    if config.optimize_anti_blaze_angle_deg:
        anti_blaze_initial = (
            initial.anti_blaze_angle_deg
            if initial.anti_blaze_angle_deg is not None
            else sum(config.anti_blaze_angle_deg_bounds.as_list()) / 2.0
        )
        anti_blaze_bounds = _resolve_bounds(
            explicit_bounds=config.anti_blaze_angle_deg_bounds,
            initial_value=float(anti_blaze_initial),
            default_relative_fraction=0.2,
        )
        parameters.append(_range_parameter("anti_blaze_angle_deg", anti_blaze_bounds))

    if config.optimize_top_cap_thickness_nm:
        top_cap_bounds = _resolve_bounds(
            explicit_bounds=config.top_cap_thickness_nm_bounds,
            initial_value=initial.top_cap_thickness_nm,
            default_relative_fraction=0.2,
            zero_default_factory=_top_cap_default_bounds,
        )
        parameters.append(_range_parameter("top_cap_thickness_nm", top_cap_bounds))

    return parameters


def build_laminar_ax_parameters(config: LaminarAxConfig) -> list[dict[str, object]]:
    """Build Ax parameter definitions from laminar optimizer config."""

    parameters: list[dict[str, object]] = []
    optimized_parameters = [
        ("period_lpermm", config.optimize_period_lpermm, config.period_lpermm_bounds),
        (
            "width_to_period_ratio",
            config.optimize_width_to_period_ratio,
            config.width_to_period_ratio_bounds,
        ),
        ("depth_nm", config.optimize_depth_nm, config.depth_nm_bounds),
        (
            "left_wall_angle_deg",
            config.optimize_left_wall_angle_deg,
            config.left_wall_angle_deg_bounds,
        ),
        (
            "right_wall_angle_deg",
            config.optimize_right_wall_angle_deg,
            config.right_wall_angle_deg_bounds,
        ),
        (
            "top_cap_thickness_nm",
            config.optimize_top_cap_thickness_nm,
            config.top_cap_thickness_nm_bounds,
        ),
        (
            "roughness_sigma_nm",
            config.optimize_roughness_sigma_nm,
            config.roughness_sigma_nm_bounds,
        ),
    ]
    for name, is_optimized, bounds in optimized_parameters:
        if is_optimized:
            parameters.append(_range_parameter(name, bounds))
    return parameters


def build_ax_parameters(config: BlazedAxConfig | LaminarAxConfig) -> list[dict[str, object]]:
    """Build Ax parameter definitions from optimizer config."""

    if isinstance(config, LaminarAxConfig):
        return build_laminar_ax_parameters(config)
    return build_blazed_ax_parameters(config)


def resolve_blazed_grating_parameters(
    config: BlazedAxConfig,
    trial_parameters: Mapping[str, float],
) -> dict[str, object]:
    """Resolve full blazed-grating kwargs from trial parameters."""

    initial: InitialBlazedGrating = config.initial_grating
    resolved_parameters: dict[str, object] = {
        "period_lpermm": float(trial_parameters.get("period_lpermm", initial.period_lpermm)),
        "blaze_angle_deg": float(trial_parameters.get("blaze_angle_deg", initial.blaze_angle_deg)),
        "anti_blaze_angle_deg": (
            float(trial_parameters["anti_blaze_angle_deg"])
            if "anti_blaze_angle_deg" in trial_parameters
            else initial.anti_blaze_angle_deg
        ),
        "substrate_material": initial.substrate_material,
        "layer_material": initial.layer_material,
        "layer_thickness_nm": initial.layer_thickness_nm,
        "top_cap_material": initial.top_cap_material,
        "top_cap_thickness_nm": float(
            trial_parameters.get("top_cap_thickness_nm", initial.top_cap_thickness_nm)
        ),
        "z_resolution_nm": initial.z_resolution_nm,
        "x_resolution_nm": initial.x_resolution_nm,
    }
    return resolved_parameters


def resolve_solver_parameters(
    config: BlazedAxConfig | LaminarAxConfig,
    trial_parameters: Mapping[str, float],
) -> dict[str, float | None]:
    """Resolve non-geometric solver parameters from trial parameters."""

    roughness_sigma_nm = trial_parameters.get(
        "roughness_sigma_nm",
        config.roughness_sigma_nm,
    )
    return {
        "roughness_sigma_nm": (
            None if roughness_sigma_nm is None else float(roughness_sigma_nm)
        ),
    }


def resolve_laminar_grating_parameters(
    config: LaminarAxConfig,
    trial_parameters: Mapping[str, float],
) -> dict[str, object]:
    """Resolve full laminar-grating kwargs from trial parameters."""

    initial: InitialLaminarGrating = config.initial_grating
    resolved_parameters: dict[str, object] = {
        "period_lpermm": float(trial_parameters.get("period_lpermm", initial.period_lpermm)),
        "width_to_period_ratio": float(
            trial_parameters.get("width_to_period_ratio", initial.width_to_period_ratio)
        ),
        "depth_nm": float(trial_parameters.get("depth_nm", initial.depth_nm)),
        "left_wall_angle_deg": float(
            trial_parameters.get("left_wall_angle_deg", initial.left_wall_angle_deg)
        ),
        "right_wall_angle_deg": float(
            trial_parameters.get("right_wall_angle_deg", initial.right_wall_angle_deg)
        ),
        "substrate_material": initial.substrate_material,
        "layer_material": initial.layer_material,
        "layer_thickness_nm": initial.layer_thickness_nm,
        "top_cap_material": initial.top_cap_material,
        "top_cap_thickness_nm": float(
            trial_parameters.get("top_cap_thickness_nm", initial.top_cap_thickness_nm)
        ),
        "z_resolution_nm": initial.z_resolution_nm,
        "x_resolution_nm": initial.x_resolution_nm,
    }
    return resolved_parameters


def resolve_grating_parameters(
    config: BlazedAxConfig | LaminarAxConfig,
    trial_parameters: Mapping[str, float],
) -> dict[str, object]:
    """Resolve full grating kwargs from trial parameters."""

    if isinstance(config, LaminarAxConfig):
        return resolve_laminar_grating_parameters(config, trial_parameters)
    return resolve_blazed_grating_parameters(config, trial_parameters)


def build_blazed_grating(
    config: BlazedAxConfig,
    trial_parameters: Mapping[str, float],
) -> BlazedGrating:
    """Build a blazed grating from Ax trial parameters."""

    return BlazedGrating(**resolve_grating_parameters(config, trial_parameters))


def build_laminar_grating(
    config: LaminarAxConfig,
    trial_parameters: Mapping[str, float],
) -> LaminarGrating:
    """Build a laminar grating from Ax trial parameters."""

    return LaminarGrating(**resolve_laminar_grating_parameters(config, trial_parameters))


def build_grating(
    config: BlazedAxConfig | LaminarAxConfig,
    trial_parameters: Mapping[str, float],
) -> BlazedGrating | LaminarGrating:
    """Build a grating from Ax trial parameters."""

    if isinstance(config, LaminarAxConfig):
        return build_laminar_grating(config, trial_parameters)
    return build_blazed_grating(config, trial_parameters)
